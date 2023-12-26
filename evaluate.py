from pathlib import Path
import shutil
import librosa

import torch
import soundfile

from modules.dc_crn import DCCRN as Model
from tools.lzf_utils.audio_utils import AudioUtils


def enchance_one(net: Model, in_wav_path: Path, out_wav_path: Path, out_input=False):
    try:
        in_data, _ = librosa.load(in_wav_path, sr=g_sr)
        inputs = torch.FloatTensor(in_data[None]).to(device)
        with torch.no_grad():
            _, output = net(inputs)
            output = output.cpu().numpy().squeeze()

        if out_input:
            output = AudioUtils.merge_channels(in_data, output)

        soundfile.write(out_wav_path, output, g_sr)
        print(out_wav_path)
    except Exception as e:
        print("error:", e)
    ...


def enhance(
    net: Model,
    idx: int,
    in_wav_list: list,
    out_wav_dir: Path,
    out_input: bool,
):
    out_wav_dir.mkdir(parents=True, exist_ok=True)
    for in_f in map(Path, in_wav_list):
        out_f = out_wav_dir.joinpath(in_f.stem + f"_ep{idx:03}" + in_f.suffix)
        enchance_one(net, in_f, out_f, out_input=out_input)
    ...


if __name__ == "__main__":
    ############ configuration start ############
    (
        frame_len,
        frame_hop,
        g_sr,
        model_index_ranges,
        device,
        out_input,
    ) = (512, 256, 16000, 21, "cpu", bool(0))
    out_wav_base_dir = Path(r"/mnt/c/AudioTest/model_test/3.out_data/tmp")
    checkpoint_dir = Path(
        r"/mnt/c/AudioTest/model_test/2.in_model/[mydesk][test]DCCRN_1225_sisdr_dnsdrb_half_16k/checkpoints/"
    )
    # in_wav_list = list(
    #     Path(r"/home/featurize/data/from_lzf/evaluation_data/4.reverb_speech/").glob(
    #         "*.wav"
    #     )
    # )
    in_wav_list = [
        # r"/mnt/c/AudioTest/model_test/1.in_data/input.wav",
        r"/mnt/c/AudioTest/model_test/1.in_data/中会议室_女声_降噪去混响测试.wav",
        r"/mnt/c/AudioTest/model_test/1.in_data/小会议室_女声_降噪去混响测试.wav",
        r"/mnt/c/AudioTest/model_test/1.in_data/大会议室_男声_降噪去混响测试_RK降噪开启.wav",
        r"/mnt/c/AudioTest/model_test/1.in_data/大会议室_男声_降噪去混响测试_RK降噪开启_mic1.wav",
        # r"/mnt/c/AudioTest/model_test/1.in_data/TB5W_工厂大会议室_去混响+非稳态降噪演示_cut00.wav",
        # r"/mnt/c/AudioTest/model_test/1.in_data/TB5W_工厂大会议室_去混响+非稳态降噪演示_cut11.wav",
        # r"/mnt/c/AudioTest/model_test/1.in_data/TB5W_工厂大会议室_去混响+非稳态降噪演示_cut22.wav",
    ]
    ############ configuration end ############

    shutil.rmtree(out_wav_base_dir, ignore_errors=True)
    if isinstance(model_index_ranges, int):
        model_index_ranges = (model_index_ranges, model_index_ranges + 1)
    for idx in range(*model_index_ranges):
        ckpt_path = checkpoint_dir.joinpath(f"model_{idx:04}.pth")
        net = Model(
            win_len=frame_len,
            win_inc=frame_hop,
            fft_len=frame_len,
        ).to(device)
        net.load_state_dict(torch.load(ckpt_path, device))
        net.eval()
        # out_wav_dir = out_wav_base_dir.joinpath(f"ep{idx:03}")
        out_wav_dir = Path(out_wav_base_dir)
        enhance(net, idx, in_wav_list, out_wav_dir, out_input)
    ...