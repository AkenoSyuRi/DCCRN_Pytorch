from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile
from time_utils import TimeUtils


def pad_or_cut(data, nsamples):
    data_len = len(data)
    if data_len < nsamples:
        data = np.pad(data, (0, nsamples - data_len), mode="wrap")
        return data

    return data[:nsamples]


def audio_sampling_down(in_wav_path, out_wav_path, tar_fs=16000, segment=10):
    try:
        data, _ = librosa.load(in_wav_path, sr=tar_fs)
        # assert len(data) == tar_fs * segment, "not a valid 10s wav file"
        nsamples = tar_fs * segment
        data = pad_or_cut(data, nsamples)
        soundfile.write(out_wav_path, data, tar_fs)
    except Exception as e:
        print(e)
        raise


@TimeUtils.measure_time
def test_multithreading(in_dir, out_dir, tar_fs=16000, segment=10):
    with ThreadPoolExecutor() as ex:
        for in_f in Path(in_dir).rglob("*.wav"):
            out_f = Path(out_dir, in_f.as_posix()[len(in_dir) :].lstrip("/"))
            out_f = out_f.parent.joinpath(out_f.stem + "_down16k.wav")
            out_f.parent.mkdir(parents=True, exist_ok=True)
            assert not out_f.exists()
            ex.submit(
                audio_sampling_down,
                in_f.as_posix(),
                out_f.as_posix(),
                tar_fs=tar_fs,
                segment=segment,
            )
    ...


if __name__ == "__main__":
    in_dir = "/home/featurize/data/from_lzf/train_data_32k/clean"
    out_dir = "/home/featurize/data/from_lzf/train_data_16k/clean"
    test_multithreading(in_dir, out_dir)
    ...
