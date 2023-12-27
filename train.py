import argparse
import os
import random

import numpy as np
import toml
import torch
from torch.utils.data import DataLoader

from dataset import Dataset_DNS, get_data_list
from tools.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DTLN")
    parser.add_argument(
        "-C",
        "--train_config",
        default="configs/train_server.toml",
        help="train configuration",
    )
    parser.add_argument(
        "-R", "--resume", action="store_true", help="Resume the experiment"
    )
    args = parser.parse_args()
    config = toml.load(args.train_config)
    print(f"config: {args.train_config}, resume: {args.resume}")

    # set random seed, and visible GPU
    seed = config["meta"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    clean_data_list = get_data_list(config["dataset"]["clean_dir_list"], shuffle=True)
    noise_data_list = get_data_list(config["dataset"]["noise_dir_list"], shuffle=True)
    rir_data_list = get_data_list(config["dataset"]["rir_dir_list"], shuffle=True)
    valid_sample_cnt = config["dataset"]["valid_sizes"]

    # create Dataset
    train_dataset = Dataset_DNS(
        clean_data_list=clean_data_list[: -valid_sample_cnt[0]],
        noise_data_list=noise_data_list[: -valid_sample_cnt[1]],
        rir_data_list=rir_data_list[: -valid_sample_cnt[2]],
        is_validation_set=False,
        samplerate=config["dataset"]["samplerate"],
        duration=config["dataset"]["duration"],
        snr_range=tuple(config["dataset"]["snr_range"]),
        scale_range=tuple(config["dataset"]["scale_range"]),
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config["meta"]["batch_size"],
        shuffle=True,
        num_workers=config["meta"]["num_workers"],
        pin_memory=config["meta"]["pin_memory"],
        drop_last=False,
    )

    valid_dataset = Dataset_DNS(
        clean_data_list=clean_data_list[-valid_sample_cnt[0] :],
        noise_data_list=noise_data_list[-valid_sample_cnt[1] :],
        rir_data_list=rir_data_list[-valid_sample_cnt[2] :],
        is_validation_set=True,
        samplerate=config["dataset"]["samplerate"],
        duration=config["dataset"]["duration"],
        snr_range=tuple(config["dataset"]["snr_range"]),
        scale_range=tuple(config["dataset"]["scale_range"]),
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=config["meta"]["batch_size"],
        shuffle=False,
        num_workers=config["meta"]["num_workers"],
        pin_memory=config["meta"]["pin_memory"],
        drop_last=False,
    )

    # create trainer and train network.
    trainer = Trainer(
        config=config,
        resume=args.resume,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
    )

    trainer.train()
    ...
