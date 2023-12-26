import time


class TimeUtils:
    @staticmethod
    def measure_time(func):
        def inner(*args, **kwargs):
            t1 = time.time()
            ret = func(*args, **kwargs)
            t2 = time.time()
            print(f"[elapsed][function:{func.__name__}]: {t2 - t1:.3f}s")
            return ret

        return inner

    @staticmethod
    def sec2hms(sec):
        h = int(sec / 3600)
        sec %= 3600

        m = int(sec / 60)
        sec %= 60

        s = sec

        hms = ""
        if h > 0:
            hms += f"{h}:" if h > 9 else f"0{h}:"

        hms += f"{m}:" if m > 9 else f"0{m}:"
        hms += f"{s:.3f}" if s > 9 else f"0{s:.3f}"

        return hms

    @staticmethod
    def hms2sec(hms):
        hms = hms.strip()
        if hms.count(":") == 1:
            hms = "0:" + hms
        h, m, s = map(float, hms.split(":"))

        sec = h * 3600 + m * 60 + s
        return round(sec, 3)
