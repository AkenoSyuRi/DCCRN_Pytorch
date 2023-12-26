import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import gpuRIR
import numpy as np
import soundfile

sys.path.append(Path(__file__).absolute().parent.parent.as_posix())
from tools.lzf_utils.file_utils import FileUtils
from tools.lzf_utils.time_utils import TimeUtils


class ReverbSimulator:
    def __init__(
        self,
        fs=32000,
        rt60_range=(0.10, 1.11),
        room_x_range=(3, 10.1),
        room_y_range=(2.5, 6.1),
        room_z_range=(2.5, 4.6),
        abs_range=(0.01, 0.41),
        att_diff_range=(5, 21),
    ):
        self.fs = fs
        self.rt60_list = np.round(np.arange(*rt60_range, 0.01), 2)
        self.room_x_list = np.round(np.arange(*room_x_range, 0.1), 1)
        self.room_y_list = np.round(np.arange(*room_y_range, 0.1), 1)
        self.room_z_list = np.round(np.arange(*room_z_range, 0.1), 1)

        # Attenuation when start using the diffuse reverberation model [dB]
        self.att_diff_list = np.arange(*att_diff_range, 1)

        self.att_max = 60.0  # Attenuation at the end of the simulation [dB]

        # Absortion coefficient ratios of the walls
        self.abs_list = np.round(np.arange(*abs_range, 0.01), 2)
        ...

    def get_rir(self):
        T60 = random.choice(self.rt60_list)  # 所需的混响时间
        room_sz = [
            random.choice(self.room_x_list),
            random.choice(self.room_y_list),
            random.choice(self.room_z_list),
        ]

        pos_src = np.array(
            [
                [
                    round(random.uniform(0.1, room_sz[0] - 0.1), 2),
                    round(random.uniform(0.1, room_sz[1] - 0.1), 2),
                    round(random.uniform(1.0, 2.1), 2),
                ]
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
            ]
        )

        abs_weights = [random.choice(self.abs_list)] * 6

        # Reflection coefficients
        beta = gpuRIR.beta_SabineEstimation(room_sz, T60, abs_weights=abs_weights)

        att_diff = random.choice(self.att_diff_list)
        # Time to start the diffuse reverberation model [s]
        Tdiff = gpuRIR.att2t_SabineEstimator(att_diff, T60)

        # Time to stop the simulation [s]
        Tmax = gpuRIR.att2t_SabineEstimator(self.att_max, T60)

        # Number of image sources in each dimension
        nb_img = gpuRIR.t2n(Tdiff, room_sz)

        # with shape: (1, 1, N1)
        RIRs = gpuRIR.simulateRIR(
            room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, self.fs, Tdiff=Tdiff
        )

        rir = RIRs.squeeze()
        rir /= np.max(np.abs(rir)) + 1e-7
        return rir, T60


def gen_rir_proc(simulator: ReverbSimulator, out_rir_path):
    rir, RT60 = simulator.get_rir()
    soundfile.write(out_rir_path % RT60, rir, simulator.fs)
    # print(out_rir_path)


@TimeUtils.measure_time
def test_multiprocessing(start_idx, generate_count, out_dir):
    FileUtils.ensure_dir(out_dir)
    simulator = ReverbSimulator()
    with ProcessPoolExecutor() as ex:
        for i in range(start_idx, start_idx + generate_count):
            out_rir_path = os.path.join(out_dir, f"gpu_rir_{i}_rt60_%.2fs.wav")
            ex.submit(gen_rir_proc, simulator, out_rir_path)
    ...


@TimeUtils.measure_time
def test_multithreading(start_idx, generate_count, out_dir):
    FileUtils.ensure_dir(out_dir)
    simulator = ReverbSimulator()
    with ThreadPoolExecutor() as ex:
        for i in range(start_idx, start_idx + generate_count):
            out_rir_path = os.path.join(out_dir, f"gpu_rir_{i}_rt60_%.2fs.wav")
            ex.submit(gen_rir_proc, simulator, out_rir_path)
    ...


if __name__ == "__main__":
    output_dir = (
        "/home/featurize/data/from_lzf/train_data_32k/rir/gpu_rir_0.1_to_1.1_cnt_10w"
    )
    test_multithreading(0, 101000, output_dir)
    ...
