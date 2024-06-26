import random
from abc import abstractmethod

import torch
from numpy import inf

from src.base import BaseModel
from src.logger import get_visualizer
from src.utils import MetricTracker


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self,
                 model,
                 criterion,
                 metrics,
                 disc_optimizer,
                 gen_optimizer,
                 disc_lr_scheduler,
                 gen_lr_scheduler,
                 config,
                 device):
        self.device = device
        self.config = config
        self.logger = config.get_logger("trainer", config["trainer"]["verbosity"])

        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.gen_optimizer = gen_optimizer,
        self.disc_optimizer = disc_optimizer,
        
        self.gen_lr_scheduler = gen_lr_scheduler
        self.disc_lr_scheduler = disc_lr_scheduler

        # for interrupt saving
        self._last_epoch = 0

        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.monitor = cfg_trainer.get("monitor", "off")
        self.log_step = cfg_trainer["log_step"]

        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = get_visualizer(config, self.logger)

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @torch.no_grad()
    def get_grad_norm(self, submodel, norm_type=2):
        parameters = submodel.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]

        total_norm = torch.norm(
            torch.stack([torch.norm(torch.nan_to_num(p.grad.detach(), nan=0), norm_type).cpu() for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError()

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__

        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "disc_optimizer": self.gen_optimizer.state_dict(),
            "gen_optimizer": self.disc_optimizer.state_dict(),
            "disc_lr_scheduler": self.disc_lr_scheduler.state_dict(),
            "gen_lr_scheduler": self.gen_lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        if not (only_best and save_best):
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        self.model.load_state_dict(checkpoint["state_dict"])

        self.logger.info("Loading optimizer state")
        print(self.disc_optimizer)
        self.disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
        self.gen_optimizer.load_state_dict(checkpoint["gen_optimizer"])

        self.logger.info("Loading scheduler state")
        self.disc_lr_scheduler.load_state_dict(checkpoint["disc_lr_scheduler"])
        self.gen_lr_scheduler.load_state_dict(checkpoint["gen_lr_scheduler"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )
        
    def _train_process(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not,
                    # according to specified metric(mnt_metric)
                    if self.mnt_mode == "min":
                        improved = log[self.mnt_metric] <= self.mnt_best
                    elif self.mnt_mode == "max":
                        improved = log[self.mnt_metric] >= self.mnt_best
                    else:
                        improved = False
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn't improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break
            
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e
