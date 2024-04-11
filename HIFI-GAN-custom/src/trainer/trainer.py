from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import itertools
import torch

from src.base import BaseTrainer
from src.utils import inf_loop, MetricTracker, DEFAULT_SR
from src.model.mel_spectrogram import MelSpectrogramConfig, MelSpectrogram
from wv_mos_metric.calculate_wv_mos import calculate_all_metrics
from wv_mos_metric.metric import MOSNet

class GANTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            disc_optimizer,
            gen_optimizer,
            disc_lr_scheduler,
            gen_lr_scheduler,
            config,
            device,
            dataloaders,
            len_epoch=None,
            skip_oom=True
    ):
        super().__init__(
            model,
            criterion,
            metrics,
            disc_optimizer,
            gen_optimizer,
            disc_lr_scheduler,
            gen_lr_scheduler,
            config,
            device
        )
        
        self.skip_oom = skip_oom
        self.config = config

        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_lr_scheduler = gen_lr_scheduler
        self.disc_lr_scheduler = disc_lr_scheduler

        mel_spec_config = MelSpectrogramConfig()
        self.mel_spec_transform = MelSpectrogram(mel_spec_config).to(self.device)

        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch

        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}

        self.loss_names = ["d_loss", "g_loss", "mpd_g_loss", "msd_g_loss", "mpd_d_loss",
                           "msd_d_loss", "mel_spec_g_loss", "mpd_features_g_loss", "msd_features_g_loss"]

        self.val_names = ['val_err', 'wv_mos_std', 'wv_mos_mean']
        self.train_metrics = MetricTracker(*self.loss_names, "Gen grad_norm", "MPDs grad_norm", "MSD grad_norm")
        self.evaluation_metrics = MetricTracker(*self.val_names)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["wav", "mel"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    @torch.no_grad()
    def _log_predictions(self, pred, wav, examples_to_log=3, **kwargs):
        rows = {}
        i = 0
        for pred, target in zip(pred, wav):
            if i >= examples_to_log:
                break
            rows[i] = {
                "pred": self.writer.wandb.Audio(pred.cpu().squeeze().numpy(), sample_rate=DEFAULT_SR),
                "target": self.writer.wandb.Audio(target.cpu().squeeze().numpy(), sample_rate=DEFAULT_SR),
            }
            i += 1

        self.writer.add_table("logs", pd.DataFrame.from_dict(rows, orient="index"))
    
    def _clip_grad_norm(self, module):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                module.parameters(), self.config["trainer"]["grad_norm_clip"]
            )
            
    @torch.no_grad()
    def _log_predictions_val(self, pred, wav, examples_to_log=3, **kwargs):
        rows = {}
        i = 0
        for pred, target in zip(pred, wav):
            if i >= examples_to_log:
                break
            rows[i] = {
                "pred_validation": self.writer.wandb.Audio(pred.cpu().squeeze().numpy(), sample_rate=DEFAULT_SR),
                "target": self.writer.wandb.Audio(target.cpu().squeeze().numpy(), sample_rate=DEFAULT_SR),
            }
            i += 1

        self.writer.add_table("logs", pd.DataFrame.from_dict(rows, orient="index"))

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        
        wav_gt, mel_gt = batch['wav'], batch['mel']
        mel_spec_gt = batch["mel"]
        
        wav_pred = self.model.generator(mel_gt)
        
        batch["pred"] = wav_pred
        mel_spec_pred = self.mel_spec_transform(wav_pred).squeeze(1)

       # -------- discriminator loss --------
        self.disc_optimizer.zero_grad()
        
        # wav_ds_mpd, wav_gen_ds_mpd, _, _ = self.model.mpd(wav_gt, wav_pred.detach())
        wav_ds_mpd, wav_gen_ds_mpd, wav_fmaps_mpd, wav_gen_fmaps_mpd = self.model.mpd(wav_gt, wav_pred.detach())
        
        wav_ds_msd, wav_gen_ds_msd, _, _ = self.model.msd(wav_gt, wav_pred.detach())

        mpd_d_loss = self.criterion.disc_adv_loss(wav_ds_mpd, wav_gen_ds_mpd)
        msd_d_loss = self.criterion.disc_adv_loss(wav_ds_msd, wav_gen_ds_msd)

        d_loss = mpd_d_loss + msd_d_loss

        d_loss.backward()
        
        self._clip_grad_norm(self.model.mpd)
        self._clip_grad_norm(self.model.msd)
        
        self.disc_optimizer.step()
        
        batch["mpd_d_loss"] = mpd_d_loss
        batch["msd_d_loss"] = msd_d_loss
        batch["d_loss"] = d_loss

        self.train_metrics.update("MPDs grad_norm", self.get_grad_norm(self.model.mpd))
        self.train_metrics.update("MSD grad_norm", self.get_grad_norm(self.model.msd))
#
        # -------- generator loss --------
        self.gen_optimizer.zero_grad()

        # wav_ds_mpd, wav_gen_ds_mpd, wav_fmaps_mpd, wav_gen_fmaps_mpd = self.model.mpd(wav_gt, wav_pred.detach())
        wav_ds_mpd, wav_gen_ds_mpd, wav_fmaps_mpd, wav_gen_fmaps_mpd = self.model.mpd(wav_gt, wav_pred.detach())
        
        wav_ds_msd, wav_gen_ds_msd, wav_fmaps_msd, wav_gen_fmaps_msd = self.model.msd(wav_gt, wav_pred.detach())

        mpd_g_loss = self.criterion.gen_adv_loss(wav_gen_ds_mpd)
        msd_g_loss = self.criterion.gen_adv_loss(wav_gen_ds_msd)

        mel_g_loss = self.criterion.mel_loss(mel_spec_gt, mel_spec_pred)

        mdp_fm_loss = self.criterion.fm_loss(wav_fmaps_mpd, wav_gen_fmaps_mpd)
        msd_fm_loss = self.criterion.fm_loss(wav_fmaps_msd, wav_gen_fmaps_msd)

      #  TODO add lambdas
        g_loss = mpd_g_loss + msd_g_loss + mel_g_loss + mdp_fm_loss + msd_fm_loss

        g_loss.backward()
        self._clip_grad_norm(self.model.generator)
        self.gen_optimizer.step()

        batch["mpd_g_loss"] = mpd_g_loss
        batch["msd_g_loss"] = msd_g_loss
        batch["mel_spec_g_loss"] = mel_g_loss
        batch["mpd_features_g_loss"] = mdp_fm_loss
        batch["msd_features_g_loss"] = msd_fm_loss
        batch["g_loss"] = g_loss

        self.train_metrics.update("Gen grad_norm", self.get_grad_norm(self.model.generator))

        for loss_name in self.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        val_err_tot = 0.0
        fake_wavs = []
        with torch.no_grad():
            for j, batch in tqdm(enumerate(dataloader), desc=part, total=len(dataloader)):
                batch = self.move_batch_to_device(batch, self.device)

                _, mel_gt = batch['wav'], batch['mel']
                wav_pred = self.model.generator(mel_gt)
                batch['pred'] = wav_pred
                fake_wavs.append(wav_pred.squeeze().cpu().numpy())
                wav_pred_mel = self.mel_spec_transform(wav_pred)
                val_err_tot += F.l1_loss(mel_gt, wav_pred_mel).item()

            val_err = val_err_tot / len(dataloader)
            scores = calculate_all_metrics(fake_wavs, [MOSNet()])
            print(scores)
            self.evaluation_metrics.update('wv_mos_mean', scores['MOSNet'][0])
            self.evaluation_metrics.update('wv_mos_std', scores['MOSNet'][1])
            self.evaluation_metrics.update('val_err', val_err)
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_predictions_val(**batch)
            self._log_scalars(self.evaluation_metrics)

        return self.evaluation_metrics.result()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc="train", total=self.len_epoch)):
            batch = self.process_batch(batch, metrics=self.train_metrics)
            try:
                batch = self.process_batch(batch, metrics=self.train_metrics)
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug("Train Epoch: {} {} Gen loss: {:.6f} Disc loss: {:.6f} Mel loss: {:.6f}".format(
                    epoch, self._progress(batch_idx), batch["g_loss"].item(), batch["d_loss"].item(), batch["mel_spec_g_loss"].item()))
                self.writer.add_scalar("disc learning rate", self.disc_lr_scheduler.get_last_lr()[0])
                self.writer.add_scalar("gen learning rate", self.gen_lr_scheduler.get_last_lr()[0])
                self._log_scalars(self.train_metrics)
                self._log_predictions(**batch)

                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()

            if batch_idx + 1 >= self.len_epoch:
                break

        self.gen_lr_scheduler.step()
        self.disc_lr_scheduler.step()
        log = last_train_metrics
        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log
