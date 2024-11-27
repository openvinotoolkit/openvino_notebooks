from argparse import ArgumentTypeError
from itertools import product
from typing import List, Literal, Optional, TypedDict


class ValidationMatrix:
    os = ("ubuntu-20.04", "ubuntu-22.04", "windows-2019", "macos-13")
    python = ("3.9", "3.10", "3.11", "3.12")
    device = ("cpu", "gpu")

    @classmethod
    def values(cls):
        return product(cls.device, cls.os, cls.python)


class ValidationConfig(TypedDict):
    os: str
    python: str
    device: str


def validation_config_arg(arg_name: Literal["os", "python", "device"]):
    available_options = getattr(ValidationMatrix, arg_name)

    def fn(value: str):
        if not value in available_options:
            raise ArgumentTypeError(f"Invalid value '{value}'. Available options: {available_options}")
        return value

    return fn


class SkipConfig(TypedDict):
    os: Optional[List[str]]
    python: Optional[List[str]]
    device: Optional[List[str]]


class SkippedNotebook(TypedDict):
    notebook: str
    skips: List[SkipConfig]
