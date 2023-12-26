import numpy as np


class DbgUtils:
    @staticmethod
    def export_npy_file(data, out_npy_path):
        np.save(out_npy_path, np.array(data))
        print("export:", out_npy_path)
