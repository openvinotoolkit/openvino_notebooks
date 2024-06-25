from itertools import product


class ValidationMatrix:
    os = ("ubuntu-20.04", "ubuntu-22.04", "windows-2019", "macos-12")
    python = ("3.8", "3.9", "3.10", "3.11")
    device = ("cpu", "gpu")

    @classmethod
    def values(cls):
        return product(cls.device, cls.os, cls.python)
