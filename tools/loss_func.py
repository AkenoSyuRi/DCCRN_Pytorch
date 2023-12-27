from functools import partial
from typing import Final

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def sisdr_loss(preds, target, zero_mean: bool = False):
    """`Scale-invariant signal-to-distortion ratio` (SI-SDR).

    The SI-SDR value is in general considered an overall measure of how good a source sound.

    Args:
        preds: float tensor with shape ``(...,time)``
        target: float tensor with shape ``(...,time)``
        zero_mean: If to zero mean target and preds or not

    Returns:
        Float tensor with shape ``(...,)`` of SDR values per sample

    Raises:
        RuntimeError:
            If ``preds`` and ``target`` does not have the same shape
    """
    eps = torch.finfo(preds.dtype).eps

    if zero_mean:
        target = target - torch.mean(target, dim=-1, keepdim=True)
        preds = preds - torch.mean(preds, dim=-1, keepdim=True)

    alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + eps) / (
        torch.sum(target**2, dim=-1, keepdim=True) + eps
    )
    target_scaled = alpha * target

    noise = preds - target_scaled

    val = (torch.sum(target_scaled**2, dim=-1) + eps) / (
        torch.sum(noise**2, dim=-1) + eps
    )
    return -10 * torch.mean(torch.log10(val))


def weighted_sisdr(mixed, clean, clean_est, eps=1e-7):
    bsum = lambda x: torch.sum(x, dim=-1)  # Batch preserving sum for convenience.
    noise = mixed - clean
    noise_est = mixed - clean_est

    a = bsum(clean**2) / (bsum(clean**2) + bsum(noise**2) + eps)
    wSISDR = a * sisdr_loss(clean_est, clean) + (1 - a) * sisdr_loss(noise_est, noise)
    return torch.mean(wSISDR)


def snr_loss(predict, clean, eps=1e-7):
    """
    Args:
        predict: Enhanced fo shape [B, T]
        clean: Reference of shape [B, T]
    Returns:
        snr: [B]
    """
    tmp = torch.mean(clean**2, dim=-1, keepdim=True) / (
        torch.mean((predict - clean) ** 2, dim=-1, keepdim=True) + eps
    )
    loss = -10 * torch.mean(torch.log10(tmp + eps))
    return loss


def weighted_snr(mixed, clean, clean_est, eps=1e-7):
    bsum = lambda x: torch.sum(x, dim=-1)  # Batch preserving sum for convenience.
    noise = mixed - clean
    noise_est = mixed - clean_est

    a = bsum(clean**2) / (bsum(clean**2) + bsum(noise**2) + eps)
    wSNR = a * snr_loss(clean_est, clean) + (1 - a) * snr_loss(noise_est, noise)
    return torch.mean(wSNR)


def wSDRLoss(mixed, clean, clean_est, eps=1e-7):
    # Used on signal level(time-domain). Backprop-able istft should be used.
    # Batched audio inputs shape (N x T) required.
    # Batch preserving sum for convenience.
    def bsum(x):
        return torch.sum(x, dim=1)

    def mSDRLoss(orig, est):
        # Modified SDR loss, <x, x`> / (||x|| * ||x`||) : L2 Norm.
        # Original SDR Loss: <x, x`>**2 / <x`, x`> (== ||x`||**2)
        #  > Maximize Correlation while producing minimum energy output.
        correlation = bsum(orig * est)
        energies = torch.norm(orig, p=2, dim=1) * torch.norm(est, p=2, dim=1)
        return -(correlation / (energies + eps))

    noise = mixed - clean
    noise_est = mixed - clean_est

    a = bsum(clean**2) / (bsum(clean**2) + bsum(noise**2) + eps)
    wSDR = a * mSDRLoss(clean, clean_est) + (1 - a) * mSDRLoss(noise, noise_est)
    return torch.mean(wSDR)


class WMPLoss(nn.Module):  # weighted phase magnitude loss
    def __init__(self, win_len=768, win_inc=256, window=None, weight=1):
        super(WMPLoss, self).__init__()
        self.weight = weight
        self.pi = torch.FloatTensor([np.pi])

        self.stft = partial(
            torch.stft,
            n_fft=win_len,
            hop_length=win_inc,
            win_length=win_len,
            window=window,
            center=False,
            return_complex=True,
        )
        ...

    def forward(self, y, y_hat):
        y_stft = self.stft(y)
        y_hat_stft = self.stft(y_hat)

        mag = torch.abs(y_stft)
        mag_hat = torch.abs(y_hat_stft)

        theta = torch.angle(y_stft)
        theta_hat = torch.angle(y_hat_stft)

        dif_theta = 2 * mag * torch.sin((theta_hat - theta) / 2)

        dif_mag = mag_hat - mag
        loss = torch.mean(dif_mag**2 + self.weight * dif_theta**2)
        return loss


def high_band_mse_loss(preds: torch.Tensor, target: torch.Tensor):
    win_len, win_inc = 512, 256
    window = torch.hamming_window(win_len, device=preds.device)

    stft = partial(
        torch.stft,
        n_fft=win_len,
        hop_length=win_inc,
        win_length=win_len,
        window=window,
        return_complex=True,
    )

    preds_stft = stft(preds)
    target_stft = stft(target)

    high_band_idx = preds_stft.shape[1] // 2

    preds_mag = torch.abs(preds_stft[:, high_band_idx:])
    target_mag = torch.abs(target_stft[:, high_band_idx:])

    loss = F.mse_loss(preds_mag, target_mag)
    return loss


def sisdr_mse_loss(clean_est, clean, alpha=100):
    return sisdr_loss(clean_est, clean) + alpha * high_band_mse_loss(clean_est, clean)


class ComplexCompressedMSELoss(nn.Module):
    gamma: Final[float]
    beta: Final[float]

    def __init__(
        self,
        gamma: float = 0.3,  # compress factor
        beta: float = 0.3,  # weighting between magnitude-based and complex loss
        win_length=1024,
        hop_length=512,
    ):
        super().__init__()
        self.gamma = gamma
        self.beta = beta

        window = torch.hann_window(
            win_length, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.stft = partial(
            torch.stft,
            n_fft=win_length,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=False,
            return_complex=True,
        )

    def forward(self, inputs, target):
        inputs_stft = self.stft(inputs)
        target_stft = self.stft(target)

        inputs_mag = torch.abs(inputs_stft).pow(self.gamma)
        target_mag = torch.abs(target_stft).pow(self.gamma)
        inputs_phase = torch.angle(inputs_stft)
        target_phase = torch.angle(target_stft)
        loss1 = torch.abs(inputs_mag - target_mag).square().mean() * (1 - self.beta)

        inputs_stft_hat = inputs_mag * torch.exp(1j * inputs_phase)
        target_stft_hat = target_mag * torch.exp(1j * target_phase)
        loss2 = torch.abs(inputs_stft_hat - target_stft_hat).square().mean() * self.beta

        loss = loss1 + loss2
        return loss
