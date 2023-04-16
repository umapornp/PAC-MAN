import json
import os
import time

from mangnn.utils.logging import LOG
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter


class CallbackCaller:
    """ Callback caller. """

    def __init__(self, callbacks, trainer):
        """ Constructor for `CallbackCaller`.
        
        Args: 
            callbacks: List of callbacks.
            trainer: Trainer.
        """
        self.callbacks = callbacks
        self.trainer = trainer
        self.set_trainer(self.trainer)


    def set_trainer(self, trainer):
        """ Set the trainer for all callbacks.
        
        Args:
            trainer: Trainer.
        """
        for callback in self.callbacks:
            callback.set_trainer(trainer)


    def on_train_begin(self):
        """ Call all Callbacks on train begin. """
        for callback in self.callbacks:
            callback.on_train_begin()


    def on_train_end(self):
        """ Call all Callbacks on train end. """
        for callback in self.callbacks:
            callback.on_train_end()


    def on_evaluate_begin(self):
        """ Call all Callbacks on evaluate begin. """
        for callback in self.callbacks:
            callback.on_evaluate_begin()


    def on_evaluate_end(self):
        """ Call all Callbacks on evaluate end. """
        for callback in self.callbacks:
            callback.on_evaluate_end()


    def on_epoch_begin(self, epoch):
        """ Call all Callbacks on epoch begin. 
        
        Args:
            epoch: Epoch.
        """
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)


    def on_epoch_end(self, epoch):
        """ Call all Callbacks on epoch end. 
        
        Args:
            epoch: Epoch.
        """
        for callback in self.callbacks:
            callback.on_epoch_end(epoch)


    def on_train_batch_begin(self, batch):
        """ Call all Callbacks on train batch begin. 
        
        Args:
            batch: Batch.
        """
        for callback in self.callbacks:
            callback.on_train_batch_begin(batch)


    def on_train_batch_end(self, batch):
        """ Call all Callbacks on train batch end. 
        
        Args:
            batch: Batch.
        """
        for callback in self.callbacks:
            callback.on_train_batch_end(batch)


    def on_evaluate_batch_begin(self, batch):
        """ Call all Callbacks on evaluate batch begin. 
        
        Args:
            batch: Batch.
        """
        for callback in self.callbacks:
            callback.on_evaluate_batch_begin(batch)


    def on_evaluate_batch_end(self, batch):
        """ Call all Callbacks on evaluate batch end. 
        
        Args:
            batch: Batch.
        """
        for callback in self.callbacks:
            callback.on_evaluate_batch_end(batch)


class Callback:
    """ Callback. """

    def __init__(self):
        """ Constructor for `Callback`. """
        self.trainer = None


    def set_trainer(self, trainer):
        """ Set trainer.
        
        Args:
            trainer: Trainer.
        """
        self.trainer = trainer


    def on_train_begin(self):
        """ On train begin. """
        pass


    def on_train_end(self):
        """ On train end. """
        pass


    def on_evaluate_begin(self):
        """ On evaluate begin. """
        pass


    def on_evaluate_end(self):
        """ On evaluate end. """
        pass


    def on_epoch_begin(self, epoch):
        """ On epoch begin.
        
        Args:
            epoch: Epoch.
        """
        pass


    def on_epoch_end(self, epoch):
        """ On epoch end. 
        
        Args:
            epoch: Epoch.
        """
        pass


    def on_train_batch_begin(self, batch):
        """ On train batch begin. 
        
        Args:
            batch: Batch.
        """
        pass


    def on_train_batch_end(self, batch):
        """ On train batch end. 
        
        Args:
            batch: Batch.
        """
        pass


    def on_evaluate_batch_begin(self, batch):
        """ On evaluate batch begin. 
        
        Args:
            batch: Batch.
        """
        pass


    def on_evaluate_batch_end(self, batch):
        """ On evaluate batch end. 
        
        Args:
            batch: Batch.
        """
        pass


class ModelCheckpoint(Callback):
    """ Save model checkpoint at the end of each epoch for every interval step. """

    def __init__(self, ckpt_path, save_every):
        """ Constructor for `ModelCheckpoint`.
        
        Args:
            ckpt_path: Checkpoint path.
            save_every: Checkpoint interval (training step).
        """
        super(ModelCheckpoint).__init__()
        self.ckpt_path = ckpt_path
        self.save_every = save_every


    def on_epoch_end(self, epoch):
        """ Save model checkpoint on epoch end. """
        if self.trainer.master_process and epoch % self.save_every == 0:
            self.trainer._save_checkpoint(epoch, self.ckpt_path)


class Timer(Callback):
    """ Timer. """

    def __init__(self):
        """ Constructor for `Timer`. """
        super(Timer).__init__()
        self.timer_train = 0.0
        self.timer_epoch = 0.0
        self.timer_batch = 0.0


    def on_train_begin(self):
        """ Start timer for training. """
        self.timer_train = self._start()


    def on_epoch_begin(self, epoch):
        """ Start timer for epoch. """
        self.timer_epoch = self._start()


    def on_train_batch_begin(self, batch):
        """" Start timer for training batch. """
        self.timer_batch = self._start()


    def on_train_end(self):
        """ Stop timer for training. """
        self.timer_train = self._stop(self.timer_train)
        LOG.info(f"Training time: {self.timer_train:0.2f} s.")
        self.timer_train = self._reset()


    def on_epoch_end(self, epoch):
        """ Stop timer for epoch. """
        self.timer_epoch = self._stop(self.timer_epoch)
        # log.info(f"Epoch time: {self.timer_epoch:0.4f} s.")
        self.timer_epoch = self._reset()


    def on_train_batch_end(self, batch):
        """ Stop timer for train batch. """
        self.timer_batch = self._stop(self.timer_batch)
        # log.info(f"Batch time: {self.timer_batch:0.4f} s.")
        self.timer_batch = self._reset()


    def _start(self):
        """ Start timer. """
        return time.time()


    def _stop(self, timer):
        """ Stop timer. 

        Args:
            timer: Timer
        """
        return time.time() - timer


    def _reset(self):
        """ Reset timer. """
        return 0.0


class Tensorboard(Callback):
    """" Tensorboard logging. """

    def __init__(self, log_dir='./logs'):
        """ Constructor for `Tensorboard`. 
        
        Args:
            log_dir: Logging directory.
        """
        super(Tensorboard, self).__init__()
        self.log_dir = log_dir


    def on_train_begin(self):
        """ Set writer and logging directory. """
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)


    def on_train_batch_end(self, batch):
        """ Log train_loss and lr. """
        self._log_loss("train_loss")
        self._log_lr()


    def on_epoch_end(self, epoch):
        """ Log val_loss and metrics. """
        self._log_loss("val_loss")
        self._log_metrics()
        LOG.info(f"Training tensorboard saved at {self.log_dir}")


    def on_train_end(self):
        """ Close writer. """
        self.writer.close()


    def _log_loss(self, name):
        """ Log loss. 
        
        Args:
            name: Type of loss. (`train_loss` and `val_loss`)
        """
        self.writer.add_scalar(name, self.trainer.losses[name].get(), self.trainer.global_steps)


    def _log_metrics(self):
        """ Log metrics. """
        for name, value in self.trainer.metrics.get().items():
            self.writer.add_scalar(name, value, self.trainer.global_steps)


    def _log_lr(self):
        """ Log lr. """
        for group, param in enumerate(self.trainer.optimizer.param_groups):
            self.writer.add_scalar(f"lr_{group}", param["lr"], self.trainer.global_steps)


class Monitor(Callback):
    """ Monitor the best losses and metrics. """

    def __init__(self, metric_name=[]):
        """ Constructor for `Monitor`. 
        
        Args:
            metric_name: List of monitoring metrics.
        """
        super(Monitor).__init__()
        self.metric_name = metric_name
        self.monitor = ["train_loss", "val_loss"] + self.metric_name 
        self.best = {}
        self._build()


    def on_epoch_end(self, epoch):
        """ Monitor the best. """
        # metrics = self.trainer.metrics.get() 
        metrics = {k:v for k,v in self.trainer.metrics.get().items() if k in self.metric_name}
        msg = ""

        # Losses.
        for name, value in self.trainer.losses.items():
            if value.get() < self.best[name]["value"]:
                self.best[name]["value"] = value.get()
                self.best[name]["epoch"] = epoch
                msg += f"{name}: {self.best[name]['value']:0.4f} | "

        # Metrics.
        for name, value in metrics.items():
            if value > self.best[name]["value"]:
                self.best[name]["value"] = value
                self.best[name]["epoch"] = epoch
                msg += f"{name}: {self.best[name]['value']:0.4f} | "

        if msg:
            msg = "New best: " + msg
            LOG.info(msg)


    def _build(self):
        """ Prepare dictionary for monitoring. """
        for name in self.monitor:
            self.best[name] = {"value":-1, "epoch":-1}


class ProgressBarLogger(Callback):
    """ Progress bar logger. """

    def __init__(self):
        """ Constructor for `ProgressBarLogger`. """
        super(ProgressBarLogger, self).__init__()
        self.pbar = None
        self.fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},{rate_inv_fmt}]{postfix}"
        self.do_train = False


    def on_train_begin(self):
        """ Log on train begin. """
        self.do_train = True
        LOG.info("Training start...")


    def on_epoch_begin(self, epoch):
        """ Log on epoch begin. """
        device = f"[{self.trainer.device}] " if self.trainer.use_ddp else ""
        LOG.info(device + f"Epoch {epoch}/{self.trainer.end_epoch}")
        self.pbar = tqdm(total=self.trainer.steps, unit="step", bar_format=self.fmt)


    def on_train_batch_end(self, batch):
        """ Log on training batch end. """
        self.pbar.set_postfix(loss=self.trainer.losses["train_loss"].get(), val_loss="?")
        self.pbar.update(1)


    def on_evaluate_begin(self):
        """ Log on evaluation batch begin. """
        if not self.do_train:
            LOG.info("Evaluation start...")


    def on_evaluate_end(self):
        """ Log on evaluation end. """
        if not self.do_train:
            self._log_loss_and_metrics(loss=["val_loss"])
            LOG.info("Evaluation finished!")


    def on_epoch_end(self, epoch):
        """ Log on epoch end. """
        self.pbar.set_postfix(loss=self.trainer.losses["train_loss"].get(), val_loss=self.trainer.losses["val_loss"].get())
        self.pbar.close()
        self._log_loss_and_metrics(loss=["train_loss", "val_loss"])


    def on_train_end(self):
        """ Log on training end. """
        LOG.info("Training finished!")


    def _log_loss_and_metrics(self, loss=[]):
        """ Log loss and metrics. 

        Args:
            loss: List of losses.
        """
        header = ""
        data = ""
        sep = "\t| "
        divider = "-"*110
        
        for name in loss:
            # name = "loss" if name == "train_loss" else name
            header += f"{name}{sep}"
            data += f"{self.trainer.losses[name].get():0.4f}{sep}"
        
        for name, value in self.trainer.metrics.get().items():
            header += f"{name}{sep}"
            data += f"{value:0.4f}{sep}"
        
        LOG.info(divider)
        LOG.info("Results")
        LOG.info(divider)
        LOG.info(header[:-len(sep)])
        LOG.info(divider)
        LOG.info(data[:-len(sep)])
        LOG.info(divider)


class JSONLogger(Callback):
    """ Save training results to JSON file. """

    def __init__(self, log_dir="./result.json"):
        """ Constructor for `JSONLogger`. 
        
        Args:
            log_dir: Logging directory.
        """
        super(JSONLogger, self).__init__()
        self.log_dir = log_dir


    def on_epoch_end(self, epoch):
        """ On epoch end. """
        # Collect epoch, train_loss, val_loss, and metrics.
        cur_logs = {"epoch": epoch, "result": {}}
        cur_logs["result"].update({k:v.get() for k,v in self.trainer.losses.items()})
        cur_logs["result"].update(self.trainer.metrics.get())

        # Append the results if the file already contains the data.
        logs = []
        if os.path.exists(self.log_dir): 
            logs = self._read()
        logs.append(cur_logs)
        self._write(logs)
        LOG.info(f"Training results saved at {self.log_dir}")


    def _write(self, data):
        """ Write data to file. 
        
        Args:
            data: List of data to write. 
        """
        with open(self.log_dir, "w") as f:
            json.dump(data, f, sort_keys=True, indent=4)


    def _read(self):
        """ Read data in file. """
        with open(self.log_dir, 'r') as f:
            data = json.load(f)
        return data