import time
from datetime import datetime
from pathlib import Path

import toml
import torch
import tqdm
from loguru import logger

from dataset import DataPrefetcher

from .early_stopping import EarlyStopping
from .initial_model import initialize_module
from .loss_func import sisdr_loss, snr_loss, wSDRLoss
from .lzf_utils.time_utils import TimeUtils


def prepare_empty_dir(dirs, resume=False):
    """
    if resume the experiment, assert the dirs exist. If not the resume experiment, set up new dirs.
    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    for dir_path in dirs:
        if resume:
            assert (
                dir_path.exists()
            ), "In resume mode, you must be have an old experiment dir."
        else:
            dir_path.mkdir(parents=True, exist_ok=True)


class Trainer:
    def __init__(self, config, resume, train_dataloader, valid_dataloader):
        model = initialize_module(config["model"]["path"], args=config["model"]["args"])
        self.config = config
        # set model, loss, optimizer, scheduler
        self.device = torch.device("cuda")
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config["optimizer"]["lr"]
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            "min",
            factor=config["scheduler"]["factor"],
            patience=config["scheduler"]["patience"],
            min_lr=config["scheduler"]["min_lr"],
            verbose=True,
        )

        self.epochs = config["meta"]["num_epochs"]

        self.start_epoch = 1
        self.save_dir = (
            Path(config["meta"]["save_model_dir"]).expanduser().absolute()
            / config["meta"]["experiment_name"]
        )
        self.checkpoints_dir = self.save_dir / "checkpoints"
        self.logs_dir = self.save_dir / "logs"
        print("save_dir:", self.save_dir)

        # clip and AMP
        self.clipset = config["trainer"]["args"]["clip_grad_norm_ornot"]
        self.clip_norm_value = config["trainer"]["args"]["clip_grad_norm_value"]

        # for realtime inference
        self.method = self.config["loss"]["method"]
        self.early_stopping = EarlyStopping(patience=7, trace_func=logger.info)

        if resume:
            self._resume_checkpoint()

        if not resume and config["meta"]["preload_model_path"]:
            self._preload_model(config["meta"]["preload_model_path"])

        prepare_empty_dir([self.checkpoints_dir, self.logs_dir], resume=resume)
        with open(
            (self.save_dir / f"{time.strftime('%Y-%m-%d %H.%M.%S')}.toml").as_posix(),
            "w",
        ) as handle:
            toml.dump(config, handle)
        logger.add(self.logs_dir / "train.log")
        self._print_networks([self.model])

    def _preload_model(self, model_path):
        """
        Preload model parameters (in "*.tar" format) at the start of experiment.

        Args:
            model_path (Path): The file path of the *.tar file
        """
        checkpoint = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        logger.info(f"[preload] Model preloaded successfully from {model_path}.")

    def _resume_checkpoint(self):
        """
        Resume the experiment from the latest checkpoint.
        """
        latest_model_path = (
            self.checkpoints_dir.expanduser().absolute() / "latest_model.tar"
        )
        assert (
            latest_model_path.exists()
        ), f"{latest_model_path} does not exist, can not load latest checkpoint."
        checkpoint = torch.load(latest_model_path.as_posix(), map_location="cpu")

        self.start_epoch = checkpoint["epoch"] + 1
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

        self.model.load_state_dict(checkpoint["model"])
        self.model.cuda()
        logger.info(
            f"[resume] Model checkpoint loaded. Training will begin at {self.start_epoch} epoch."
        )

    def _save_checkpoint(self, epoch):
        """
        Save checkpoint to "<save_dir>/<config name>/checkpoints" directory, which consists of:
            - epoch
            - optimizer parameters
            - model parameters
        """
        logger.info(f"\t Saving {epoch} epoch model checkpoint...")
        state_dict = {
            "epoch": epoch,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "model": self.model.state_dict(),
        }
        torch.save(
            state_dict["model"],
            (self.checkpoints_dir / f"model_{str(epoch).zfill(4)}.pth").as_posix(),
        )
        torch.save(
            state_dict, (self.checkpoints_dir / "latest_model.tar").as_posix()
        )  # Latest

    @staticmethod
    def _print_networks(models: list):
        print(
            f"This project contains {len(models)} models, the number of the parameters is: "
        )
        params_of_all_networks = 0
        for idx, model in enumerate(models, start=1):
            params_of_network = 0
            for param in model.parameters():
                if not param.requires_grad:
                    continue
                params_of_network += param.numel()
            print(f"\tNetwork {idx}: {params_of_network / 1e6} million.")
            params_of_all_networks += params_of_network
        print(
            f"The amount of parameters in the project is {params_of_all_networks / 1e6} million."
        )

    def calloss(self, wavTrue, wavPred, wavMixed, method="wSDR"):
        if method == "wSDR":
            return wSDRLoss(wavMixed, wavTrue, wavPred)

        if method == "snr":
            return snr_loss(wavPred, wavTrue)

        if method == "sisdr":
            return sisdr_loss(wavPred, wavTrue)

        raise "pls set right loss"

    @staticmethod
    def check_nan(loss):
        if torch.isnan(loss).any():
            msg = "NAN encountered in loss"
            logger.error(msg)
            raise ValueError(msg)

    def _train_epoch_ns(self, epoch, num_epochs):
        loss = 0

        progress_bar = tqdm.tqdm(total=len(self.train_dataloader), desc="Training")
        prefetcher = DataPrefetcher(self.train_dataloader, self.device)
        for batch_id, (mixAudio, cleanAudio) in enumerate(prefetcher, start=1):
            # self.scheduler.step(epoch-1)
            self.optimizer.zero_grad()

            _, output = self.model(mixAudio)

            # calculate loss and update parameters
            nloss = self.calloss(
                wavPred=output,
                wavTrue=cleanAudio,
                wavMixed=mixAudio,
                method=self.method,
            )
            self.check_nan(nloss)
            nloss.backward()
            if self.clipset:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.clip_norm_value, norm_type=2
                )
            self.optimizer.step()
            # self.scheduler.step()
            loss += nloss.item()
            # train log
            progress_bar.update(1)
            progress_bar.refresh()
            if batch_id % 100 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"\n[{datetime.now()}]"
                    f"Train epoch [{epoch} / {num_epochs}],"
                    f"batch: [{batch_id} / {len(self.train_dataloader)}],"
                    f"loss[{self.method}]: {(loss / (batch_id)):.5f},"
                    f"lr: {lr:.5f}"
                    "\n"
                )
            del mixAudio, cleanAudio, output, nloss

    def _validation(self, epoch):
        loss = cnt = 0

        progress_bar = tqdm.tqdm(total=len(self.valid_dataloader), desc="Validating")
        prefetcher = DataPrefetcher(self.valid_dataloader, self.device)
        with torch.no_grad():
            for _, (mixAudio, cleanAudio) in enumerate(prefetcher, start=1):
                _, output = self.model(mixAudio)

                # calculate loss and update parameters
                nloss = self.calloss(
                    wavPred=output,
                    wavTrue=cleanAudio,
                    wavMixed=mixAudio,
                    method=self.method,
                )

                loss += nloss.item()
                # train log
                progress_bar.update(1)
                progress_bar.refresh()
                cnt += 1
                del mixAudio, cleanAudio, output, nloss
        val_loss = loss / cnt
        lr = self.optimizer.param_groups[0]["lr"]
        msg = (
            f"\n[{datetime.now()}][Validation]"
            f"loss[{self.method}]: {val_loss:2.6f},"
            f"lr: {lr:.5f}"
            "\n"
        )
        logger.info(msg)

        old_best = self.scheduler.best
        self.scheduler.step(val_loss)
        new_best = self.scheduler.best
        if old_best <= new_best:
            logger.info(
                f"Rejected !!! The best is {new_best:2.6f}, but we get {val_loss:2.6f}"
            )
        else:
            logger.info(
                f"Accepted !!! Get the best [epoch/loss]: {epoch}/{new_best:2.6f}"
            )
        return val_loss

    @TimeUtils.measure_time
    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            print("[0 seconds]Begin training...")
            self.model.train()
            self._train_epoch_ns(epoch, self.epochs)
            self.model.eval()

            val_loss = self._validation(epoch)
            self._save_checkpoint(epoch)
            torch.cuda.empty_cache()  # release the memory of which tensors is deleted manully

            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                logger.info("early stopping triggered, break the training loop")
                break
