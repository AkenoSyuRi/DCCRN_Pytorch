import re
import os
import sys

sys.path.append(os.path.abspath(".."))

from tools.lzf_utils.file_utils import FileUtils
from tools.lzf_utils.time_utils import TimeUtils

if __name__ == "__main__":
    in_dir = r"/home/featurize/data/from_lzf/rir/gpu_rir"
    rt60_pat = re.compile("_rt60_(\d+\.\d+)s")
    
    result_map = {
        "0.1": 0, "0.2": 0, "0.3": 0, "0.4": 0, "0.5": 0, "0.6": 0, "0.7": 0, "0.8": 0, "0.9": 0, "1.0": 0, 
        "1.1": 0, "1.2": 0, "1.3": 0, "1.4": 0, "1.5": 0,
    }
    for idx, in_wav_path in enumerate(FileUtils.iglob_files(f"{in_dir}/*.wav")):
        rt60 = rt60_pat.search(in_wav_path).group(1)[:3]
        assert rt60 in result_map
        result_map[rt60] += 1
        # print(f"{rt60:.2f} {in_wav_path}")
        
    print(result_map)
    ...