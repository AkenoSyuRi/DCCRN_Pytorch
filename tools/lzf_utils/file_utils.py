import glob
import os
import random
import re
from itertools import chain


class FileUtils:
    @staticmethod
    def name_ext(file_path):
        basename = os.path.basename(file_path)
        name, ext = os.path.splitext(basename)
        return name, ext[1:]

    @staticmethod
    def ensure_dir(dir_or_file, *, is_file=False):
        if is_file:
            dir_or_file = os.path.dirname(dir_or_file)

        if not os.path.exists(dir_or_file):
            os.makedirs(dir_or_file)

    @staticmethod
    def name_sub(filepath, *sub_patterns, base_dir=None, splitter="#"):
        if base_dir:
            filename = os.path.basename(filepath)
        else:
            filename = filepath

        for sub_pat in sub_patterns:
            assert (
                splitter in sub_pat
            ), f"invalid sub_pattern: {sub_pat}, splitter is missing: {splitter}"
            from_pat, to_pat = sub_pat.split(splitter)
            filename = re.sub(from_pat, to_pat, filename)

        if base_dir:
            new_filepath = os.path.join(base_dir, filename)
        else:
            new_filepath = filename
        return new_filepath

    @staticmethod
    def iglob_files(*patterns):
        it_list = []
        for pat in patterns:
            recursive = "**" in pat
            it_list.append(glob.iglob(pat, recursive=recursive))
        return chain(*it_list)

    @classmethod
    def glob_files(cls, *patterns, shuffle=False):
        it = cls.iglob_files(*patterns)
        files = list(it)
        if shuffle:
            random.shuffle(files)
        return files
