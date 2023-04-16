import os
import torch

from mangnn.utils.logging import TrainLogger, LOG
from mangnn.utils.callbacks import CallbackCaller
from mangnn.utils.metrics import MetricCaller

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler
from torch.amp import autocast


class Trainer:
    """ Trainer for training the model. """

    def __init__(
        self,
        model,
        begin_epoch,
        end_epoch,
        optimizer,
        lr_scheduler,
        criterion,
        metrics,
        callbacks,
        device="cpu",
        use_amp=True,
        gradient_accumulate_steps=1,
        clip_grad=1.0,
        resume_from="",
        dtype="bfloat16"
    ):
        """ Constructor for `Trainer`.
        
        Args:
            model: Model to train.
            begin_epoch: Beginning of the training epoch.
            end_epoch: End of the training epoch.
            optimizer: Optimizer.
            lr_scheduler: Learning rate scheduler.
            criterion: Loss function.
            metrics: Evaluation metrics.
            callbacks: Callbacks.
            device: Running device. (Default=`cpu`).
            use_amp: Whether to use Automatic Mixed Precision (AMP). (Default=`True`).
            gradient_accumulate_steps: Gradient accumulation steps. 1 is disable. (Default=`1`).
            clip_grad: Gradient clipping. 0.0 is disable. (Default=`1.0`).
            resume_from: Path of training model to resume from.
            dtype: Data type. (Default=`bfloat16`).
        """
        # Set rank (for DDP training).
        self.local_rank = int(os.environ.get('LOCAL_RANK', -1))
        self.global_rank = int(os.environ.get('RANK', -1))
        self.use_ddp = self.local_rank != -1

        # set master process.
        if self.use_ddp:
            self.master_process = self.global_rank == 0
        else:
            self.master_process = True

        # Set device.
        self.device_type = device
        if self.local_rank != -1:
            self.device = f"{self.device_type}:{self.local_rank}"
        else:
            self.device = self.device_type

        # Set model.
        self.raw_model = model
        self._model = None
        
        # Set data type.
        self.dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

        # Set optimizer and learning rate.
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Set training epoch.
        self.begin_epoch = begin_epoch
        self.end_epoch = end_epoch

        # Set AMP.
        self.use_amp = use_amp 
        if self.device_type != "cuda" and self.use_amp: # Available for GPU only.
            self.use_amp = False 
            LOG.info(f"Automatically disable AMP for {self.device_type} device.")
        self.scaler = GradScaler(enabled=self.use_amp)

        # Set gradient clipping.
        self.clip_grad = clip_grad
        
        # Set gradient accumulation steps.
        self.gradient_accumulate_steps = gradient_accumulate_steps

        # Set loss function and logger.
        self.criterion = criterion
        self.losses = {k: TrainLogger(k) for k in ["train_loss", "val_loss"]}
        
        # Set metrics.
        self.metrics = MetricCaller(metrics=metrics)
        
        # Set callbacks.
        self.callbacks = CallbackCaller(callbacks=callbacks, trainer=self)

        # Set training steps.
        self.global_steps = -1
        self.steps = -1

        # Resume from checkpoint.
        self.resume_from = resume_from 
        if self.resume_from:
            self._load_checkpoint()

        # Set model device. 
        self.raw_model.to(self.device)


    @property
    def model(self):
        """ Return the model. """
        if self._model is None:
            self._build_model()
        return self._model


    def _build_model(self):
        """ Wrap the model (if DDP is enabled).
        
        Returns:
            Model or wrapped model (if DDP is enabled).
        """
        if self.use_ddp:
            if self.device_type == "cuda":  
                self._model = DDP(self.raw_model, device_ids=[self.local_rank])
            else:
                self._model = DDP(self.raw_model)
        else:
            self._model = self.raw_model

        return self._model


    def _train_step(self, inputs, labels, step):
        """" Training step.
        
        Args:
            inputs: Dictionary of user_ids, text_ids, and tag_ids.
            labels: Tag labels.
            step: Current step.
        """
        # Clear the paramter gradients.
        self.optimizer.zero_grad(set_to_none=True)
    
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        labels = labels.to(self.device)

        self.model.train()
        self.criterion.to(self.device)
        self.criterion.train()

        if self.use_ddp:
            self.model.require_backward_grad_sync = (step == self.gradient_accumulate_steps - 1)

        # Forward.
        with autocast(device_type=self.device_type, dtype=self.dtype, enabled=self.use_amp):
            logits = self.model(**inputs).logits
            loss = self.criterion(logits, labels)

        if self.gradient_accumulate_steps > 1:
            self._accumulate_grad(loss)

        # Backward.
        self.scaler.scale(loss).backward()
        
        # Update loss for logging.
        self.losses['train_loss'].update(loss)

        if (step + 1) % self.gradient_accumulate_steps == 0 or (step + 1 == self.steps):
            # Clip gradient.
            if self.clip_grad != 0.0:
                self._clip_grad()

            # Update parameters.
            self.scaler.step(self.optimizer)

            # Update scaler.
            old_scaler = self.scaler.get_scale()
            self.scaler.update()

            # Update lr.
            if self.lr_scheduler and not isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self._lr_scheduler(old_scaler=old_scaler)


    def _evaluate_step(self, inputs, labels):
        """ Evaluation step. 
        
        Args:
            inputs: Dictionary of user_ids, text_ids, and tag_ids.
            labels: Tag labels.
        """
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        labels = labels.to(self.device)
        self.model.eval()

        with torch.no_grad():
            logits = self.model(**inputs).logits # Evaluate.
            loss = self.criterion(logits, labels) # Compute val_loss.

        # Update val_loss and metrics.
        self.losses['val_loss'].update(loss)
        self.metrics.update(logits, labels)


    def _train(self, dataloader, epoch):
        """ Training for one epoch. 
        
        Args:
            dataloader: Training dataloader. 
            epoch: Current epoch. 
        """
        # Reset loss for new epoch.
        self.losses["train_loss"].reset()

        for step, (inputs, labels) in enumerate(dataloader):
            self.global_steps = ((epoch-1) * self.steps) + (step+1)
            self.callbacks.on_train_batch_begin(step)
            self._train_step(inputs, labels, step)
            self.callbacks.on_train_batch_end(step)


    def _evaluate(self, dataloader):
        """ Evaluation for one epoch. 
        
        Args:
            dataloader: Validation dataloader.
        """
        # Reset loss and metrics for new epoch.
        self.losses["val_loss"].reset()
        self.metrics.reset()

        for nbatch, (inputs, labels) in enumerate(dataloader):
            self.callbacks.on_evaluate_batch_begin(nbatch)
            self._evaluate_step(inputs, labels)
            self.callbacks.on_evaluate_batch_end(nbatch)
        
        # Compute metrics for current epoch.
        self.metrics.compute()


    def train(self, train_loader, val_loader):
        """ Training. 
        
        Args:
            train_loader: Training dataloader.
            val_loader: Validation dataloader.
        
        Returns:
            Dictionary of train_loss, val_loss, metrics.
        """
        self.steps = len(train_loader)
        self.callbacks.on_train_begin()

        for epoch in range(self.begin_epoch, self.end_epoch + 1):
            self.callbacks.on_epoch_begin(epoch)

            # Train.
            self._train(train_loader, epoch)
            
            # Evaluate.
            self.evaluate(val_loader)
            
            # Update lr (for ReduceLROnPlateau scheduler).
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self._lr_scheduler(val_loss=self.losses["val_loss"].get())

            self.callbacks.on_epoch_end(epoch)

        self.callbacks.on_train_end()

        # Return train_loss, val_loss, and metrics.
        outputs = {k:v.get() for k,v in self.losses.items()} 
        outputs.update(self.metrics.get())

        return outputs


    def evaluate(self, dataloader):
        """ Evaluation. 
        
        Args:
            dataloader: Dataloader.
        
        Returns:
            Dictionary of val_loss and metrics.
        """
        self.callbacks.on_evaluate_begin()
        self._evaluate(dataloader)
        self.callbacks.on_evaluate_end()

        # Return val_loss and metrics.
        outputs = {"val_loss": self.losses["val_loss"].get()}
        outputs.update(self.metrics.get())
        return outputs


    def _lr_scheduler(self, old_scaler=None, val_loss=None):
        """ Learning rate scheduler.
        
        Args:
            old_scaler: Old scaler.
            val_loss: Validation loss.
        
        """
        # Update lr based on val_loss (ReduceLROnPlateau).
        if val_loss:
            self.lr_scheduler.step(val_loss)
        
        # Update lr only when optimizer.step() is not skipped.
        else:
            new_scaler = self.scaler.get_scale()
            if new_scaler >= old_scaler:
                self.lr_scheduler.step()


    def _accumulate_grad(self, loss):
        """ Accumulate the gradient.
        
        Args:
            loss: Loss. 
        """
        loss = loss / self.gradient_accumulate_steps


    def _clip_grad(self):
        """ Clip the gradient. """
        # Unscale the gradient.
        self.scaler.unscale_(self.optimizer)

        # Clip the gradient.
        clip_grad_norm_(self.model.parameters(), self.clip_grad)


    def _save_checkpoint(self, epoch, ckpt_path):
        """ Save model checkpoint. 
        
        Args:
            epoch: Epoch.
            ckpt_path: Checkpoint path.
        """
        save_model = self.model.module if self.use_ddp else self.model
        checkpoint = {
            "MODEL": save_model.state_dict(),
            "OPTIMIZER": self.optimizer.state_dict(),
            "EPOCH": epoch,
            "GLOBAL_STEPS": self.global_steps,
            "LOSS": self.losses["val_loss"].get()
        }

        if self.master_process and not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint path {ckpt_path} not found.")

        filename = os.path.join(ckpt_path, 'ckpt.pt')
        torch.save(checkpoint, filename)
        LOG.info(f"Training snapshot saved at {ckpt_path}")


    def _load_checkpoint(self):
        """ Load model checkpoint. """

        if not os.path.exists(self.resume_from):
            raise FileNotFoundError(f"Checkpoint path {self.resume_from} not found.")

        device = self.device if self.device_type == 'cuda' else 'cpu'
        checkpoint = torch.load(self.resume_from, map_location=device)

        self.raw_model.load_state_dict(checkpoint["MODEL"])
        self.optimizer.load_state_dict(checkpoint['OPTIMIZER'])
        self.begin_epoch = checkpoint["EPOCH"]
        self.global_steps = checkpoint["GLOBAL_STEPS"]
        self.losses["val_loss"].update(checkpoint["LOSS"])

        LOG.info(f"Resuming training from {self.resume_from} at Epoch-{self.begin_epoch} (Step-{self.global_steps}) with val_loss {self.losses['val_loss'].get():0.4f}.")