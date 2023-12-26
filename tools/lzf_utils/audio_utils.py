import os.path
import wave

import librosa
import numpy as np
import soundfile


class AudioUtils:
    @staticmethod
    def ffmpeg_convert(
        in_audio_path, out_audio_path, sr=32000, nchannels=1, overwrite=True
    ):
        extra_flags = "-y" if overwrite else "-n"
        extra_flags += " -v error"
        cmd = f"ffmpeg -i {in_audio_path} -ar {sr} -ac {nchannels} -acodec pcm_s16le {extra_flags} {out_audio_path}"
        ret = os.system(cmd)
        if ret == 0:
            print(out_audio_path)

    @staticmethod
    def apply_gain(sig, inc_db, rms_ref=1.0):
        """
        increase the volume of `sig` by `inc_db`
        :param sig: signal of floating point type in range [-1, 1]
        :param inc_db: the relative increment in db, can be positive or negative
        :param rms_ref: 1.0 for floating point
        :return: the new signal after volume up
        """
        assert np.issubdtype(
            sig.dtype, np.floating
        ), "sig is not of floating-point type"

        cur_rms = np.sqrt(np.mean(sig**2))
        cur_db = 20 * np.log10(cur_rms / rms_ref)
        tar_db = cur_db + inc_db

        tar_rms = 10 ** (tar_db / 20)
        tar_sig = sig * tar_rms / cur_rms  # the signal is proportional to rms amplitude
        return tar_sig

    @staticmethod
    def save_to_mono(data, sr, base_dir, name):
        assert data.ndim == 1 or data.ndim == 2

        if data.ndim == 1:
            data = data.reshape(1, -1)
            channel_first = True
        else:
            channel_first = data.shape[0] < data.shape[1]

        if channel_first:
            n_channels = data.shape[0]
            if n_channels == 1:
                out_wav_path = os.path.join(base_dir, f"{name}.wav")
                soundfile.write(out_wav_path, data[0], sr)
            elif n_channels == 2:
                l_out_wav_path = os.path.join(base_dir, f"{name}_L.wav")
                r_out_wav_path = os.path.join(base_dir, f"{name}_R.wav")
                soundfile.write(l_out_wav_path, data[0], sr)
                soundfile.write(r_out_wav_path, data[1], sr)
            else:
                for i in range(n_channels):
                    out_wav_path = os.path.join(base_dir, f"{name}_chn{i}.wav")
                    soundfile.write(out_wav_path, data[i], sr)
        else:
            n_channels = data.shape[1]
            if n_channels == 1:
                out_wav_path = os.path.join(base_dir, f"{name}.wav")
                soundfile.write(out_wav_path, data[:, 0], sr)
            elif n_channels == 2:
                l_out_wav_path = os.path.join(base_dir, f"{name}_L.wav")
                r_out_wav_path = os.path.join(base_dir, f"{name}_R.wav")
                soundfile.write(l_out_wav_path, data[:, 0], sr)
                soundfile.write(r_out_wav_path, data[:, 1], sr)
            else:
                for i in range(n_channels):
                    out_wav_path = os.path.join(base_dir, f"{name}_chn{i}.wav")
                    soundfile.write(out_wav_path, data[:, i], sr)
        ...

    @staticmethod
    def save_to_segment(data, sr, win_len, win_sft, base_dir, name):
        data_len = len(data)
        if data_len < win_len:
            clip = np.pad(data, (0, win_len - data_len), mode="wrap")
            out_wav_path = os.path.join(base_dir, f"{name}_seg0.wav")
            soundfile.write(out_wav_path, clip, sr)
            print(out_wav_path)
            return

        for i, j in enumerate(range(0, data_len, win_sft)):
            out_wav_path = os.path.join(base_dir, f"{name}_seg{i}.wav")
            clip = data[j : j + win_len]

            nsamples = len(clip)
            if nsamples < win_len:
                if nsamples < (win_len // 4):
                    break
                clip = np.pad(clip, (0, win_len - nsamples), mode="wrap")
                # break

            soundfile.write(out_wav_path, clip, sr)
            print(out_wav_path)

    @staticmethod
    def data_generator(in_audio_path, frame_time, *, sr=None, ret_bytes=False):
        data, _ = librosa.load(in_audio_path, sr=sr)
        frame_len = int(sr * frame_time)

        for i in range(0, len(data), frame_len):
            clip = data[i : i + frame_len]
            if len(clip) == frame_len:
                if ret_bytes:
                    clip = (clip * 32768).astype(np.short)
                    yield clip.tobytes()
                else:
                    yield clip
        ...

    @staticmethod
    def wav_data_generator(in_audio_path, frame_time, *, sr=None, ret_bytes=False):
        assert in_audio_path.endswith(".wav"), "support wav format only"
        frame_len = int(sr * frame_time)

        with wave.Wave_read(in_audio_path) as fp:
            assert fp.getframerate() == sr
            assert fp.getnchannels() == 1
            assert fp.getsampwidth() == 2

            desired_buff_len = frame_len * 2
            buff = fp.readframes(frame_len)
            while len(buff) == desired_buff_len:
                clip = np.frombuffer(buff, dtype=np.short)
                if ret_bytes:
                    yield clip.tobytes()
                else:
                    yield clip / 32768
                buff = fp.readframes(frame_len)

    @staticmethod
    def merge_channels(*data_list):
        n_channels = len(data_list)

        assert n_channels > 1

        max_len = 0
        for i in range(n_channels):
            assert data_list[i].ndim == 1
            max_len = max(data_list[i].shape[0], max_len)

        out_data = np.zeros((max_len, n_channels), dtype=data_list[0].dtype)
        for i in range(n_channels):
            data_len = data_list[i].shape[0]
            out_data[:data_len, i] = data_list[i]

        return out_data

    @staticmethod
    def pcm2wav(pcm_path, wav_path, sample_rate=32000, n_channels=1, sample_width=2):
        assert not os.path.exists(wav_path)

        with open(pcm_path, "rb") as fp1, wave.Wave_write(wav_path) as fp2:
            raw_data = fp1.read()
            fp2.setsampwidth(sample_width)
            fp2.setnchannels(n_channels)
            fp2.setframerate(sample_rate)
            fp2.writeframes(raw_data)

    @staticmethod
    def wav2pcm(wav_path, pcm_path, overwrite=False):
        if not overwrite:
            assert not os.path.exists(pcm_path)

        with open(pcm_path, "wb") as fp1, wave.Wave_read(wav_path) as fp2:
            raw_data = fp2.readframes(fp2.getnframes())
            fp1.write(raw_data)
