import torch
from scipy import signal


class SimpleSTFT:
    def __init__(self, frame_len=1024, frame_hop=256, window=None, device=None):
        super(SimpleSTFT, self).__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.device = device

        if window != "none":
            if window == "hann":
                _window = signal.get_window(window, frame_len)
            elif window == "sin":
                _window = signal.get_window("hann", frame_len) ** 0.5
            else:
                _window = signal.get_window(window, frame_len)
            self.window = torch.from_numpy(_window).float().to(device)
        else:
            self.window = torch.ones(frame_len, device=device)

        self.win_sum = self.get_win_sum()
        ...

    def get_win_sum(self):
        window, win_len, win_inc = self.window, self.frame_len, self.frame_hop
        assert win_len % win_inc == 0, "win_len cannot be equally divided by win_inc"

        win_square = window**2
        win_hop = win_len - win_inc
        win_tmp = torch.zeros(win_len + win_hop, device=self.device)

        loop_cnt = win_len // win_inc
        for i in range(loop_cnt):
            win_tmp[i * win_inc : i * win_inc + win_len] += win_square
        win_sum = win_tmp[win_hop : win_hop + win_inc]
        return win_sum[0]  # values in this array are all the same

    def transform(self, x, return_complex=False):
        """
        in: [B,T']
        out: [B,F,T]
        """
        y = torch.stft(
            x,
            n_fft=self.frame_len,
            hop_length=self.frame_hop,
            win_length=self.frame_len,
            window=self.window,
            center=False,
            return_complex=True,
        )
        if return_complex:
            return y
        mag = torch.abs(y)
        phase = torch.angle(y)
        return mag, phase

    def inverse(self, x, transpose=False):
        """
        in: [B,F,T]
        out: [B,T']
        """
        if transpose:
            x = torch.transpose(x, 1, 2)
        y = torch.istft(
            x,
            n_fft=self.frame_len,
            hop_length=self.frame_hop,
            win_length=self.frame_len,
            window=self.window,
            center=False,
            return_complex=False,
        )
        return y


if __name__ == "__main__":
    from icecream import ic
    import torchaudio

    win_len, win_inc, window = 768, 256, "hamming"

    in_clean_path = r"F:\BaiduNetdiskDownload\BZNSYP\Wave\007537.wav"
    inputs, sr = torchaudio.load(in_clean_path)
    stft = SimpleSTFT(win_len, win_inc, window=window, device="cpu")
    ic(inputs.shape)
    mag, phase = stft.transform(inputs)
    ic(mag.shape)
    spec = mag * torch.exp(1j * phase)
    output = stft.inverse(spec)
    ic(output.shape)
    torchaudio.save("a.wav", output, sr)

    # inputs = torch.rand(2, 320000)
    # stft = SimpleSTFT(win_len, win_inc, window=window)
    # mag, phase = stft.transform(inputs)
    # ic(mag.shape)
    # spec = mag * torch.exp(1j * phase)
    # output = stft.inverse(spec)
    # ic(output.shape)
    ...
