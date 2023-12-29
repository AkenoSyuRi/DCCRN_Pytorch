import os
import random
import time

import librosa
import numpy as np
import scipy.signal as ss
import soundfile
import toml
import torch
from torch.utils import data

from tools.lzf_utils.file_utils import FileUtils


class DataPrefetcher:
    def __init__(self, loader, device="cuda"):
        self.batch = None
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch

        if batch is None:
            raise StopIteration

        self.preload()
        return batch

    def __iter__(self):
        return self


class AudioAugment:
    def __init__(
        self,
        fs=32000,
        shift_range=(0.05, 0.41),
        speed_range=(0.5, 1.6),
        step_range=(-5, 6),
    ):
        self.fs = fs
        self.shift_list = (np.arange(*shift_range, 0.02) * fs).astype(int)
        self.speed_list = np.round(np.arange(*speed_range, 0.05), 2)
        self.step_list = np.setdiff1d(np.arange(*step_range, 1), [0])
        ...

    def time_shift(self, x, shift_index: int = None):
        shift = shift_index or random.choice(self.shift_list)
        return np.roll(x, shift)

    def time_stretch(self, x, speed: float = None):
        rate = speed or random.choice(self.speed_list)
        return librosa.effects.time_stretch(x, rate=rate)

    def pitch_shifting(self, x, step: float = None):
        n_step = step or random.choice(self.step_list)
        return librosa.effects.pitch_shift(
            x, sr=self.fs, n_steps=n_step, bins_per_octave=12
        )

    @staticmethod
    def freq_mask(
        x,
        num_mask=1,
        mask_percentage=0.015,
        win_len=1024,
        win_inc=512,
        window="hamming",
    ):
        x_stft = librosa.stft(
            x, n_fft=win_len, hop_length=win_inc, win_length=win_len, window=window
        )
        fft_bins = x_stft.shape[0]
        mask_width = int(mask_percentage * fft_bins)
        for i in range(num_mask):
            mask_start = np.random.randint(low=0, high=fft_bins - mask_width)
            x_stft[mask_start : mask_start + mask_width :] = 0
        x_masked = librosa.istft(
            x_stft, n_fft=win_len, hop_length=win_inc, win_length=win_len, window=window
        )
        return x_masked


def scale_to_ref(tar_data, ref_data, eps=1e-7):
    ref_rms = np.sqrt(np.mean(ref_data**2))
    cur_rms = np.sqrt(np.mean(tar_data**2)) + eps
    return tar_data / cur_rms * ref_rms


def get_truncated_rir(rir, sr, *, direct_range=(-0.001, 0.08)):
    rir_early = np.zeros_like(rir)

    peak_idx = np.argmax(np.abs(rir))
    start_idx = max(0, peak_idx + int(sr * direct_range[0]))
    end_idx = peak_idx + int(sr * direct_range[1])

    rir[:start_idx] = 0

    rir_early[start_idx:end_idx] = rir[start_idx:end_idx]

    rir_early = scale_to_ref(rir_early, rir)
    return rir_early


def get_rts_rir(
    rir,
    sr,
    *,
    original_T60: float = 1.0,
    target_T60=0.45,
    direct_range=(-0.001, 0.02),
):
    assert rir.ndim == 1, "rir must be a 1D array."

    q = 3 / (target_T60 * sr) - 3 / (original_T60 * sr)

    peak_idx = np.argmax(np.abs(rir))
    start_idx = max(0, peak_idx + int(sr * direct_range[0]))
    end_idx = peak_idx + int(sr * direct_range[1])

    rir[:start_idx] = 0

    win = np.zeros_like(rir)
    win[start_idx:end_idx] = 1
    win[end_idx:] = 10 ** (-q * np.arange(rir.shape[0] - end_idx))
    rts_rir = rir * win

    rts_rir = scale_to_ref(rts_rir, rir)
    return rts_rir


def get_decayed_and_attenuated_rir(
    rir, sr, *, direct_range=(-0.001, 0.02), rd=0.2, t1=0.03, alpha=0.4
):
    # get decayed and attenuated function
    t = np.arange((len(rir)))
    t0 = int(sr * direct_range[1])
    t1 = int(sr * t1)
    rd = int(sr * rd)

    y1 = 10 ** (-3 * (t - t0) / rd)
    y1[:t0] = 1

    y2 = (1 + alpha) / 2 + (1 - alpha) / 2 * np.cos(np.pi * (t - t0) / (t1 - t0))
    y2[:t0] = 1
    y2[t1:] = alpha

    y = y1 * y2

    # apply function
    peak_idx = np.argmax(np.abs(rir))
    start_idx = max(0, peak_idx + int(sr * direct_range[0]))

    rir[:start_idx] = 0

    target_rir = rir.copy()
    target_rir[peak_idx:] *= y[:-peak_idx] if peak_idx else y

    target_rir = scale_to_ref(target_rir, rir)
    return target_rir


def get_data_list(dir_list: list, shuffle=True):
    for d in dir_list:
        assert os.path.exists(d), d

    files = FileUtils.glob_files(
        *map(lambda x: os.path.join(x, "**/*.[mw][pa][3v]"), dir_list), shuffle=shuffle
    )
    return files


def pad_or_cut(data, nsamples):
    data_len = len(data)
    if data_len < nsamples:
        data = np.pad(data, (0, nsamples - data_len), mode="wrap")
        return data

    return data[:nsamples]


def compute_scale_by_snr(ori_data, ref_data, snr, eps):
    # calculate the rms amplitude of ori and ref signal
    ori_rms = np.sqrt(np.mean(ori_data**2)) + eps
    ref_rms = np.sqrt(np.mean(ref_data**2)) + eps

    # Calculate the desired clean signal rms amplitude based on SNR
    tar_rms = ref_rms * (10 ** (snr / 20))

    # Scale the clean signal to the desired RMS amplitude
    scale = tar_rms / ori_rms
    return scale


def audio_mix(
    clean_data,
    noise_data,
    rir_data,
    snr,
    scale,
    /,
    sr,
    segment_length,
    add_clean=True,
    add_noise=True,
    add_rir=True,
    eps=1e-7,
):
    def add_reverb(clean, rir):
        # rir_label = get_truncated_rir(rir, sr)
        rir_label = get_rts_rir(rir, sr)
        # rir_label = get_decayed_and_attenuated_rir(rir, sr)

        clean_echoic = ss.convolve(clean, rir)[:segment_length]
        label_echoic = ss.convolve(clean, rir_label)[:segment_length]
        return clean_echoic, label_echoic

    if add_clean and add_noise and add_rir:
        clean_echoic, label_echoic = add_reverb(clean_data, rir_data)

        factor = compute_scale_by_snr(clean_echoic, noise_data, snr, eps)
        clean_echoic *= factor
        label_echoic *= factor

        noisy = clean_echoic + noise_data
        label = label_echoic
    # elif add_clean and add_noise:
    #     factor = compute_scale_by_snr(clean_data, noise_data, snr, eps)
    #     clean_data *= factor

    #     noisy = clean_data + noise_data
    #     label = clean_data
    # elif add_clean and add_rir:
    #     noisy, label = add_reverb(clean_data, rir_data)
    # elif add_clean:
    #     noisy = clean_data
    #     label = noisy.copy()
    # elif add_noise:
    #     noisy = noise_data
    #     label = np.zeros_like(noisy)
    else:
        raise NotImplementedError

    # scale the signal into the scale_range
    scale /= np.max(np.abs([noisy, label])) + eps
    noisy *= scale
    label *= scale

    return noisy, label


class Dataset_DNS(data.Dataset):
    def __init__(
        self,
        clean_data_list,
        noise_data_list,
        rir_data_list,
        is_validation_set=False,
        samplerate=32000,
        duration=10,
        snr_range=(-5, 30, 1),
        scale_range=(0.10, 0.99, 0.05),
    ):
        super().__init__()
        self.is_validation_set = is_validation_set
        self.sr = samplerate
        self.segment_length = duration * samplerate
        self.snr_list = np.arange(*snr_range)
        self.scale_list = np.arange(*scale_range)
        # self.rt60_pat = re.compile(r"_rt60_(\d+\.\d+)s")
        # self.augment = AudioAugment(samplerate)

        self.clean_data_list = clean_data_list
        self.noise_data_list = noise_data_list
        self.rir_data_list = rir_data_list

        self.clean_length = len(self.clean_data_list)
        self.noise_length = len(self.noise_data_list)
        self.rir_length = len(self.rir_data_list)

        set_type = "valid" if is_validation_set else "train"
        print(f"[{__class__.__name__}][{set_type}] clean_length:", self.clean_length)
        print(f"[{__class__.__name__}][{set_type}] noise_length:", self.noise_length)
        print(f"[{__class__.__name__}][{set_type}] rir_length:", self.rir_length)
        ...

    def __getitem__(self, index):
        if self.is_validation_set:
            return self.process_valid(index)
        return self.process_train(index)

    def process_train(self, index):
        # 10% clean, 10% noise, 15% clean+noise, 15% clean*rir, 50% clean*rir+noise
        add_clean = add_noise = add_rir = True

        # prob = np.random.rand()
        # if prob < 0.1:
        #     add_noise = add_rir = False
        # elif prob < 0.2:
        #     add_clean = add_rir = False
        # elif prob < 0.35:
        #     add_rir = False
        # elif prob < 0.5:
        #     add_noise = False

        if add_clean:
            clean_data_path = self.clean_data_list[index]
            clean_data, _ = librosa.load(clean_data_path, sr=self.sr)
            clean_data = pad_or_cut(clean_data, self.segment_length)
        else:
            clean_data = None

        if add_noise:
            noise_data_path = random.choice(self.noise_data_list)
            noise_data, _ = librosa.load(noise_data_path, sr=self.sr)
            noise_data = pad_or_cut(noise_data, self.segment_length)
        else:
            noise_data = None

        if add_rir:
            # assert add_clean, "when add_rir is True, add_clean need to be True"
            rir_data_path = random.choice(self.rir_data_list)
            rir_data, _ = librosa.load(rir_data_path, sr=self.sr)
        else:
            rir_data = None

        scale = random.choice(self.scale_list)
        snr = random.choice(self.snr_list)
        noisy_data, label_data = audio_mix(
            clean_data,
            noise_data,
            rir_data,
            snr,
            scale,
            self.sr,
            self.segment_length,
            add_clean,
            add_noise,
            add_rir,
        )

        noisy_data = noisy_data.astype(np.float32)
        label_data = label_data.astype(np.float32)
        return noisy_data, label_data

    def process_valid(self, index):
        add_clean = add_noise = add_rir = True

        # 10% clean, 10% noise, 15% clean+noise, 15% clean*rir, 50% clean*rir+noise
        # add_clean = (1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)[
        #     index % 20
        # ]
        # add_noise = (0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1)[
        #     index % 20
        # ]
        # add_rir = (0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1)[
        #     index % 20
        # ]

        # 1. clean data
        if add_clean:
            clean_data_path = self.clean_data_list[index]
            clean_data, _ = librosa.load(clean_data_path, sr=self.sr)
            clean_data = pad_or_cut(clean_data, self.segment_length)
        else:
            clean_data = None

        # 2. noise data
        if add_noise:
            noise_data_path = self.noise_data_list[index % self.noise_length]
            noise_data, _ = librosa.load(noise_data_path, sr=self.sr)
            noise_data = pad_or_cut(noise_data, self.segment_length)
        else:
            noise_data = None

        # 3. rir data
        if add_rir:
            # assert add_clean, "when add_rir is True, add_clean need to be True"
            rir_data_path = self.rir_data_list[index % self.rir_length]
            rir_data, _ = librosa.load(rir_data_path, sr=self.sr)
        else:
            rir_data = None

        scale = self.scale_list[index % len(self.scale_list)]
        snr = self.snr_list[index % len(self.snr_list)]
        noisy_data, label_data = audio_mix(
            clean_data,
            noise_data,
            rir_data,
            snr,
            scale,
            self.sr,
            self.segment_length,
            add_clean,
            add_noise,
            add_rir,
        )

        noisy_data = noisy_data.astype(np.float32)
        label_data = label_data.astype(np.float32)
        return noisy_data, label_data

    def __len__(self):
        return self.clean_length


if __name__ == "__main__":
    config = toml.load("./configs/train_server.toml")
    seed = config["meta"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    clean_data_list = get_data_list(config["dataset"]["clean_dir_list"], shuffle=True)
    noise_data_list = get_data_list(config["dataset"]["noise_dir_list"], shuffle=True)
    rir_data_list = get_data_list(config["dataset"]["rir_dir_list"], shuffle=True)
    valid_sample_cnt = config["dataset"]["valid_sizes"]

    is_validation_set = bool(0)
    if is_validation_set:
        clean_data_list = clean_data_list[-valid_sample_cnt[0] :]
        noise_data_list = noise_data_list[-valid_sample_cnt[1] :]
        rir_data_list = rir_data_list[-valid_sample_cnt[2] :]
    else:
        clean_data_list = clean_data_list[: -valid_sample_cnt[0]]
        noise_data_list = noise_data_list[: -valid_sample_cnt[1]]
        rir_data_list = rir_data_list[: -valid_sample_cnt[2]]

    train_dataset = Dataset_DNS(
        clean_data_list=clean_data_list,
        noise_data_list=noise_data_list,
        rir_data_list=rir_data_list,
        is_validation_set=is_validation_set,
        samplerate=config["dataset"]["samplerate"],
        duration=config["dataset"]["duration"],
        snr_range=tuple(config["dataset"]["snr_range"]),
        scale_range=tuple(config["dataset"]["scale_range"]),
    )

    # ================ #
    test_out_dir = config["dataset"]["test_out_wav_dir"]
    FileUtils.ensure_dir(test_out_dir)

    print("do")
    t1 = time.time()
    for i in range(100):
        noisy_data, clean_data = train_dataset[i]
        soundfile.write(
            os.path.join(test_out_dir, f"{i}_a_noisy.wav"),
            noisy_data,
            samplerate=config["dataset"]["samplerate"],
        )
        soundfile.write(
            os.path.join(test_out_dir, f"{i}_b_clean.wav"),
            clean_data,
            samplerate=config["dataset"]["samplerate"],
        )
        print(i)
    print(f"elapsed: {time.time() - t1:.3f}s")
    print("done")
