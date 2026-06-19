# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import functools
import logging
import os
import site
import sys
from collections.abc import Callable
from itertools import chain
from pathlib import Path
from typing import Optional

import openvino
from openvino.utils.node_factory import NodeFactory

from .__version__ import __version__


logger = logging.getLogger(__name__)


_ext_name = "openvino_tokenizers"
if sys.platform == "win32":
    _ext_name = f"{_ext_name}.dll"
    _bin_dir = "Release"
elif sys.platform == "darwin":
    _ext_name = f"lib{_ext_name}.dylib"
    _bin_dir = "Release"
elif sys.platform == "linux":
    _ext_name = f"lib{_ext_name}.so"
    _bin_dir = ""
else:
    sys.exit(f"Error: extension does not support the platform {sys.platform}")

# when the path to the extension set manually
_extension_path = os.environ.get("OV_TOKENIZER_PREBUILD_EXTENSION_PATH")
_openvino_dir = os.environ.get("OpenVINO_DIR")
if _extension_path and Path(_extension_path).is_file():
    # when the path to the extension set manually
    _ext_path = Path(_extension_path)
else:
    _openvino_path = []
    if _openvino_dir:
        try:
            # extension binary from OpenVINO installation dir has higher priority
            _system_type = next((Path(_openvino_dir).parent / "lib").iterdir(), Path()).name
            _openvino_path = [Path(_openvino_dir).parent / "lib" / _system_type / _bin_dir]
        except FileNotFoundError:
            logger.debug(f"Skip OpenVINO_DIR because is not OpenVINO installation path: {_openvino_dir}")

    site_packages = chain(
        _openvino_path, (Path(__file__).parent.parent,), site.getusersitepackages(), site.getsitepackages()
    )
    _ext_path = next(
        (
            ext
            for site_package in map(Path, site_packages)
            if (ext := site_package / __name__ / "lib" / _ext_name).is_file()
        ),
        _ext_name,  # Case when the library can be found in the PATH/LD_LIBRAY_PATH
    )

logger.debug(f"OpenVINO Tokenizers extension path: {_ext_path}")

del _ext_name

is_openvino_tokenizers_compatible = True
_compatibility_message = (
    "OpenVINO and OpenVINO Tokenizers versions are not binary compatible.\n"
    f"OpenVINO version:            {openvino.get_version()}\n"
    f"OpenVINO Tokenizers version: {__version__}\n\n"
    "Try to reinstall OpenVINO Tokenizers with from PyPI:\n"
    "Release version: "
    "pip install -U openvino openvino_tokenizers\n"
    "Nightly version: "
    "pip install --pre -U openvino openvino_tokenizers --extra-index-url "
    "https://storage.openvinotoolkit.org/simple/wheels/nightly"
)


@functools.lru_cache(1)
def _check_openvino_binary_compatibility() -> None:
    global is_openvino_tokenizers_compatible, _compatibility_message
    _core = openvino.Core()
    try:
        _core.add_extension(str(_ext_path))
        is_openvino_tokenizers_compatible = True
    except RuntimeError:
        is_openvino_tokenizers_compatible = False
        logger.warning(_compatibility_message)


_check_openvino_binary_compatibility()

# patching openvino
old_core_init = openvino.Core.__init__
old_factory_init = openvino.utils.node_factory.NodeFactory.__init__
old_fe_init = openvino.frontend.frontend.FrontEnd.__init__


@functools.wraps(old_core_init)
def new_core_init(self, *args, **kwargs):
    old_core_init(self, *args, **kwargs)
    self.add_extension(str(_ext_path))  # Core.add_extension doesn't support Path object


@functools.wraps(old_factory_init)
def new_factory_init(self, *args, **kwargs):
    old_factory_init(self, *args, **kwargs)
    self.add_extension(_ext_path)


@functools.wraps(old_fe_init)
def new_fe_init(self, *args, **kwargs):
    old_fe_init(self, *args, **kwargs)
    self.add_extension(str(_ext_path))


def get_create_wrapper(old_create: Callable) -> Callable:
    @functools.wraps(old_fe_init)
    def new_create(*args, **kwargs):
        op_name = args[0] if len(args) > 0 else None
        if len(args) > 0 and op_name in ["StringTensorUnpack", "StringTensorPack"]:
            msg = f"Creating {op_name} from extension is deprecated. Consider creating operation from original opset factory."
            f'E.g. _get_opset_factory("opset15").create("{op_name}", ...)'
            logger.info(msg)
        return old_create(*args, **kwargs)

    return new_create


if is_openvino_tokenizers_compatible:
    openvino.Core.__init__ = new_core_init
    openvino.frontend.frontend.FrontEnd.__init__ = new_fe_init


def _get_factory_callable() -> Callable[[], NodeFactory]:
    factory = {}

    def inner(opset_version: Optional[str] = None) -> NodeFactory:
        nonlocal factory
        if opset_version not in factory:
            if is_openvino_tokenizers_compatible:
                openvino.utils.node_factory.NodeFactory.__init__ = new_factory_init
            factory[opset_version] = NodeFactory() if opset_version is None else NodeFactory(opset_version)
            if opset_version is None:
                factory[opset_version].create = get_create_wrapper(factory[opset_version].create)

        return factory[opset_version]

    return inner


def _get_opset_factory_callable() -> Callable[[], NodeFactory]:
    # factory without extensions
    factory = {}

    def inner(opset_version: Optional[str] = None) -> NodeFactory:
        nonlocal factory
        if opset_version not in factory:
            openvino.utils.node_factory.NodeFactory.__init__ = old_factory_init
            factory[opset_version] = NodeFactory() if opset_version is None else NodeFactory(opset_version)

        return factory[opset_version]

    return inner


_get_factory = _get_factory_callable()
_get_opset_factory = _get_opset_factory_callable()

# some files uses _get_factory function
from .build_tokenizer import build_rwkv_tokenizer  # noqa
from .convert_tokenizer import convert_tokenizer  # noqa
from .utils import add_greedy_decoding, connect_models  # noqa
