import re


class StringUtils:
    @staticmethod
    def camel_case(s):
        s = re.sub(r"[._-]+", " ", s).title().replace(" ", "")
        return "".join([s[0].lower(), s[1:]])
