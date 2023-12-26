import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rir_generator
import soundfile

sys.path.append(Path(__file__).absolute().parent.parent.as_posix())
from tools.lzf_utils.file_utils import FileUtils
from tools.lzf_utils.time_utils import TimeUtils


class ReverbSimulator:
    def __init__(
        self,
        fs=32000,
        sound_velocity=340,
        n_receiver=1,
        rt60_range=(0.10, 1.11),
        room_x_range=(3, 11.1),
        room_y_range=(3, 6.1),
        room_z_range=(3, 4.6),
    ):
        self.fs = fs
        self.rt60_list = np.round(np.arange(*rt60_range, 0.01), 2)
        self.room_x_list = np.round(np.arange(*room_x_range, 0.1), 1)
        self.room_y_list = np.round(np.arange(*room_y_range, 0.1), 1)
        self.room_z_list = np.round(np.arange(*room_z_range, 0.1), 1)

        self.c = sound_velocity
        self.n_receiver = n_receiver
        ...

    def get_rir(self):
        rt60 = random.choice(self.rt60_list)  # 所需的混响时间
        room_sz = [
            random.choice(self.room_x_list),
            random.choice(self.room_y_list),
            random.choice(self.room_z_list),
        ]

        pos_src = np.array(
            [
                round(random.uniform(0.1, room_sz[0] - 0.1), 2),
                round(random.uniform(0.1, room_sz[1] - 0.1), 2),
                round(random.uniform(1.0, 2.1), 2),
            ]
        )

        # 房间中放置mic
        pos_rcv = np.array(
            [
                [
                    round(random.uniform(0.1, room_sz[0] - 0.1), 2),
                    round(random.uniform(0.1, room_sz[1] - 0.1), 2),
                    round(random.uniform(1.0, room_sz[2] - 0.1), 2),
                ]
                for _ in range(self.n_receiver)
            ]
        )

        rir = rir_generator.generate(
            self.c,
            self.fs,
            pos_rcv,
            pos_src,
            room_sz,
            reverberation_time=rt60,
            nsample=int(rt60 * self.fs),
        )

        rir /= np.max(np.abs(rir), axis=0) + 1e-7
        return rir, rt60


def gen_rir_proc(simulator: ReverbSimulator, out_rir_path):
    rir, rt60 = simulator.get_rir()
    for i in range(simulator.n_receiver):
        prefix, ext = os.path.splitext(out_rir_path % rt60)
        save_path = prefix + f"_p{i}" + ext
        soundfile.write(save_path, rir[:, i], simulator.fs)
        # print(out_rir_path)


@TimeUtils.measure_time
def test_multiprocessing(start_idx, generate_count, out_dir):
    FileUtils.ensure_dir(out_dir)
    simulator = ReverbSimulator()
    with ProcessPoolExecutor() as ex:
        for i in range(start_idx, start_idx + generate_count):
            out_rir_path = os.path.join(out_dir, f"rir_gen_{i}_rt60_%.2fs.wav")
            ex.submit(gen_rir_proc, simulator, out_rir_path)
    ...


@TimeUtils.measure_time
def test_multithreading(start_idx, generate_count, out_dir):
    FileUtils.ensure_dir(out_dir)
    simulator = ReverbSimulator()
    with ThreadPoolExecutor() as ex:
        for i in range(start_idx, start_idx + generate_count):
            out_rir_path = os.path.join(out_dir, f"rir_gen_{i}_rt60_%.2fs.wav")
            ex.submit(gen_rir_proc, simulator, out_rir_path)
    ...


if __name__ == "__main__":
    output_dir = (
        "/home/featurize/data/from_lzf/train_data_32k/rir/rir_gen_0.1_to_1.1_cnt_1w"
    )
    test_multithreading(0, 11000, output_dir)
    # test_multiprocessing(100, 100, output_dir)
    ...
