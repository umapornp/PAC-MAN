import argparse
import torch

from pacbert.dataset import TwitterDataset
from pacbert.model import PACBertForRecommendation
from pacbert.trainer import Trainer
from pacbert.utils.logging import LOG
from pacbert.utils.datamodule import TwitterDataModule
from pacbert.utils.callbacks import ModelCheckpoint, ProgressBarLogger, Monitor, Timer, Tensorboard, JSONLogger
from pacbert.utils.metrics import Precision, Recall, F1Score
from pacbert.utils.optimizer import WarmupLinearSchedule, WarmupCosineSchedule, group_optimizer_parameters
from pacbert.utils.loss import CrossEntropyHungarianLoss
from mangnn.graph import GraphLoader

from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributed import init_process_group, destroy_process_group


def parse_args():
    """ Arguments for running the model.
    
    Returns:
        args: Arguments.
    """
    parser = argparse.ArgumentParser('Run Model')

    parser.add_argument('--cfg', type=str, default='pacbert/config/config.yaml', 
                        help='Path to config')
    parser.add_argument('--dist', default=False, action='store_true',
                        help='Whether to use distributed training')

    return parser.parse_args()


def setup_config(args):
    """ Read config and merge with CLI arguments.
    
    Args:
        args: Arguments.
    
    Returns:
        config: Configuration.
    """
    if args.cfg is not None:
        config = OmegaConf.load(args.cfg)

    config.USE_DDP = args.dist

    return config


def setup_device(config):
    """ Setup device. 
    
    Args:
        config: Configuration.
    
    Returns:
        device: Device name. 
    """    
    # Determine the training device.
    if config.DEVICE == 'cpu':
        device = f'cpu'

    elif config.DEVICE == 'cuda':
        device = f'cuda'

    elif config.DEVICE == 'auto':
        device = f'cuda' if torch.cuda.is_available() else f'cpu'

    LOG.info(f"Running on device {device}.")

    return device


def setup_data(config):
    """ Create and prepare dataset. 
    
    Args: 
        config: Configuration.
    
    Returns:
        datamodule: Datamodule.
    """
    # Create dataset.
    train_dataset = TwitterDataset(split="train")
    val_dataset = TwitterDataset(split="val")
    test_dataset = TwitterDataset(split="test")

    # Prepare dataset.
    datamodule = TwitterDataModule(config.BATCH_SIZE, config.NUM_WORKERS, config.USE_DDP)
    datamodule.train_dataset = train_dataset
    datamodule.val_dataset = val_dataset
    datamodule.test_dataset = test_dataset

    return datamodule


def setup_model(config, device):
    """ Setup model. 
    
    Args:
        config: Configuration.
    
    Returns:
        model: Model.
    """
    # Initialize from BERT pretrained model and MANGNN pretrained model.
    if config.INIT_TYPE == "pretrained":

        gnn_config, gnn_graphs = None, None
        if config.GNN_PATH and config.GNN_CONFIG:
            gnn_config = OmegaConf.load(config.GNN_CONFIG)
            gnn_graphs = GraphLoader(gnn_config.NETWORK_PATH).graphs
            gnn_config = gnn_config.MANGNN
            
        model = PACBertForRecommendation.from_pretrained(bert_name=config.INIT_FROM,
                                                         config=config.PACBERT,
                                                         gnn_path=config.GNN_PATH,
                                                         gnn_config=gnn_config,
                                                         gnn_graphs=gnn_graphs,
                                                         device=device,
                                                         map_gnn_params=True,
                                                         use_last_gnn=True)

    # Initialize from scratch or checkpoint.
    else:
        model = PACBertForRecommendation(config.PACBERT)

    return model


def setup_optimizer(config, model):
    """ Setup optimizer. 
    
    Args:
        config: Configuration.
        model: Model.
    
    Returns:
        optimizer: Optimizer.
    """
    # Group model parameters.
    optim_groups = group_optimizer_parameters(model=model,
                                                weight_decay=config.WEIGHT_DECAY)

    # Set optimizer.
    if config.OPTIMIZER == "AdamW":
        optimizer = AdamW(optim_groups, 
                        lr=config.LR,
                        betas=eval(config.BETAS))

    return optimizer


def setup_lr_scheduler(config, optimizer, train_loader):
    """ Setup learning rate. 
    
    Args:
        config: Configuration.
        optimizer: Optimizer.
        train_loader: Training dataloader.
    
    Returns:
        lr_scheduler: LR Scheduler.
    """
    t_total = int(config.END_EPOCH * len(train_loader) / config.GRADIENT_ACCUMULATION_STEPS)
    last_epoch = int((config.BEGIN_EPOCH-1) * len(train_loader) / config.GRADIENT_ACCUMULATION_STEPS) - 1

    if config.LR_SCHEDULER == "plateau":
        lr_scheduler = ReduceLROnPlateau(optimizer,
                                        mode='max',
                                        factor=config.LR_FACTOR,
                                        patience=1,
                                        threshold=1e-4,
                                        cooldown=2,
                                        verbose=True,)

    elif config.LR_SCHEDULER == "warmup_linear":
        lr_scheduler = WarmupLinearSchedule(optimizer=optimizer,
                                            warmup_steps=config.WARMUP_STEPS,
                                            t_total=t_total,
                                            last_epoch=last_epoch)

    elif config.LR_SCHEDULER == "warmup_cosine":
        lr_scheduler = WarmupCosineSchedule(optimizer=optimizer,
                                            warmup_steps=config.WARMUP_STEPS,
                                            t_total=t_total,
                                            last_epoch=last_epoch)

    return lr_scheduler


def setup_loss():
    """ Setup loss function. 
    
    Returns:
        loss: Loss function. 
    """
    loss = CrossEntropyHungarianLoss()
    return loss


def setup_metrics(config):
    """ Setup evaluation metrics.
    
    Args:
        config: Configuration.
    
    Returns:
        metrics: List of evaluation metrics. 
    """
    metrics = []
    for k in config.TOPK:
        metrics.append(Precision(num_classes=config.PACBERT.tag_size, topk=k))
        metrics.append(Recall(num_classes=config.PACBERT.tag_size, topk=k))
        metrics.append(F1Score(num_classes=config.PACBERT.tag_size, topk=k))
    return metrics


def setup_callbacks(config, metrics):
    """ Setup callbacks.
    
    Args: 
        config: Configuration.
        metrics: List of evaluation metrics.
    
    Returns:
        callbacks: List of callbacks.
    """
    ckpt = ModelCheckpoint(ckpt_path=config.CKPT_PATH, save_every=config.SAVE_EVERY)
    pbar = ProgressBarLogger()
    time = Timer()
    mnt = Monitor(metric_name=[m.name for m in metrics])
    tsb = Tensorboard(log_dir=config.TSB_DIR)
    json = JSONLogger(log_dir=config.JSON_DIR)

    callbacks = [pbar, mnt, ckpt, json, tsb, time]
    return callbacks


def main(config):
    """ Main process. """

    # Setup device.
    device = setup_device(config)

    # Pre-processing for DDP.
    if config.USE_DDP:
        backend = "nccl" if device == "cuda" else "gloo"
        init_process_group(backend=backend)

    # Setup.
    datamodule = setup_data(config)
    model = setup_model(config, device)
    optimizer = setup_optimizer(config, model)
    lr_scheduler = setup_lr_scheduler(config, optimizer, datamodule.train_loader)
    loss = setup_loss()
    metrics = setup_metrics(config)
    callbacks = setup_callbacks(config, metrics)
    resume_from = config.INIT_FROM if config.INIT_TYPE == "checkpoint" else ""

    # Prepare trainer.
    trainer = Trainer(
        model=model,
        begin_epoch=config.BEGIN_EPOCH,
        end_epoch=config.END_EPOCH,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=loss,
        metrics=metrics,
        callbacks=callbacks,
        device=device,
        use_amp=config.USE_AMP,
        gradient_accumulate_steps=config.GRADIENT_ACCUMULATION_STEPS,
        clip_grad=config.CLIP_GRAD,
        resume_from=resume_from,
        dtype=config.DTYPE
    )

    # Train.
    if config.DO_TRAIN:
        trainer.train(datamodule.train_loader, datamodule.val_loader)

    # Evaluate.
    if config.DO_TEST:
        trainer.evaluate(datamodule.test_loader)

    # Post-processing for DDP.
    if config.USE_DDP:
        destroy_process_group()


if __name__ == '__main__':
    args = parse_args()
    config = setup_config(args)
    main(config)
