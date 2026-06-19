#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import functools
import importlib.util
import logging
import operator as op
import sys
from collections import OrderedDict
from typing import Union

from packaging.version import Version, parse


if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


logger = logging.getLogger(__name__)

STR_OPERATION_TO_FUNC = {">": op.gt, ">=": op.ge, "==": op.eq, "!=": op.ne, "<=": op.le, "<": op.lt}

_optimum_version = importlib_metadata.version("optimum")
_optimum_intel_version = importlib_metadata.version("optimum-intel")

_transformers_available = importlib.util.find_spec("transformers") is not None
_transformers_version = "N/A"
if _transformers_available:
    try:
        _transformers_version = importlib_metadata.version("transformers")
    except importlib_metadata.PackageNotFoundError:
        _transformers_available = False

_tokenizers_available = importlib.util.find_spec("tokenizers") is not None
_tokenizers_version = "N/A"
if _tokenizers_available:
    try:
        _tokenizers_version = importlib_metadata.version("tokenizers")
    except importlib_metadata.PackageNotFoundError:
        _tokenizers_available = False

_torch_available = importlib.util.find_spec("torch") is not None
_torch_version = "N/A"
if _torch_available:
    try:
        _torch_version = importlib_metadata.version("torch")
    except importlib_metadata.PackageNotFoundError:
        _torch_available = False


_openvino_available = importlib.util.find_spec("openvino") is not None
_openvino_version = "N/A"
if _openvino_available:
    try:
        from openvino import get_version

        version = get_version()
        # avoid invalid format
        if "-" in version:
            ov_major_version, dev_info = version.split("-", 1)
            commit_id = dev_info.split("-")[0]
            version = f"{ov_major_version}-{commit_id}"
        _openvino_version = version
    except ImportError:
        _openvino_available = False

_nncf_available = importlib.util.find_spec("nncf") is not None
_nncf_version = "N/A"
if _nncf_available:
    try:
        _nncf_version = importlib_metadata.version("nncf")
    except importlib_metadata.PackageNotFoundError:
        _nncf_available = False

_diffusers_available = importlib.util.find_spec("diffusers") is not None
_diffusers_version = "N/A"
if _diffusers_available:
    try:
        _diffusers_version = importlib_metadata.version("diffusers")
    except importlib_metadata.PackageNotFoundError:
        _diffusers_available = False


_open_clip_available = importlib.util.find_spec("open_clip") is not None
_open_clip_version = "N/A"
if _open_clip_available:
    try:
        _open_clip_version = importlib_metadata.version("open_clip_torch")
    except importlib_metadata.PackageNotFoundError:
        pass


_huggingface_hub_available = importlib.util.find_spec("huggingface_hub") is not None
_huggingface_hub_version = "N/A"
if _huggingface_hub_available:
    try:
        _huggingface_hub_version = importlib_metadata.version("huggingface_hub")
    except importlib_metadata.PackageNotFoundError:
        _huggingface_hub_available = False


_safetensors_version = "N/A"
_safetensors_available = importlib.util.find_spec("safetensors") is not None
if _safetensors_available:
    try:
        _safetensors_version = importlib_metadata.version("safetensors")
    except importlib_metadata.PackageNotFoundError:
        _safetensors_available = False


_timm_available = importlib.util.find_spec("timm") is not None
_timm_version = "N/A"
if _timm_available:
    try:
        _timm_version = importlib_metadata.version("timm")
    except importlib_metadata.PackageNotFoundError:
        _timm_available = False


_datasets_available = importlib.util.find_spec("datasets") is not None
_datasets_version = "N/A"

if _datasets_available:
    try:
        _datasets_version = importlib_metadata.version("datasets")
    except importlib_metadata.PackageNotFoundError:
        _datasets_available = False

_pillow_available = importlib.util.find_spec("PIL") is not None
if _pillow_available:
    try:
        importlib_metadata.version("pillow")
    except importlib_metadata.PackageNotFoundError:
        _pillow_available = False


_accelerate_available = importlib.util.find_spec("accelerate") is not None
_accelerate_version = "N/A"

if _accelerate_available:
    try:
        _accelerate_version = importlib_metadata.version("accelerate")
    except importlib_metadata.PackageNotFoundError:
        _accelerate_available = False

_numa_available = importlib.util.find_spec("numa") is not None

if _numa_available:
    try:
        importlib_metadata.version("py-libnuma")
    except importlib_metadata.PackageNotFoundError:
        _numa_available = False


_psutil_available = importlib.util.find_spec("psutil") is not None

if _psutil_available:
    try:
        importlib_metadata.version("psutil")
    except importlib_metadata.PackageNotFoundError:
        _psutil_available = False


_sentence_transformers_available = importlib.util.find_spec("sentence_transformers") is not None
_sentence_transformers_version = "N/A"
if _sentence_transformers_available:
    try:
        _sentence_transformers_version = importlib_metadata.version("sentence_transformers")
    except importlib_metadata.PackageNotFoundError:
        _sentence_transformers_available = False


_langchain_hf_available = importlib.util.find_spec("langchain_huggingface") is not None
_langchain_hf_version = "N/A"
if _langchain_hf_available:
    try:
        _langchain_hf_version = importlib.metadata.version("langchain_huggingface")
    except importlib.metadata.PackageNotFoundError:
        _langchain_hf_available = False


def is_transformers_available():
    return _transformers_available


def is_tokenizers_available():
    return _tokenizers_available


def is_openvino_available():
    return _openvino_available


@functools.lru_cache(1)
def is_openvino_tokenizers_available():
    if not is_openvino_available():
        return False

    if importlib.util.find_spec("openvino_tokenizers") is None:
        logger.info(
            "OpenVINO Tokenizers is not available. To deploy models in production "
            "with C++ code, please follow installation instructions: "
            "https://github.com/openvinotoolkit/openvino_tokenizers?tab=readme-ov-file#installation\n"
        )
        return False

    try:
        pip_metadata_version = importlib_metadata.version("openvino")
    except importlib_metadata.PackageNotFoundError:
        pip_metadata_version = False
    try:
        pip_metadata_version = importlib_metadata.version("openvino-nightly")
        is_nightly = True
    except importlib_metadata.PackageNotFoundError:
        is_nightly = False

    try:
        import openvino_tokenizers

        openvino_tokenizers._get_factory()
    except RuntimeError:
        tokenizers_version = openvino_tokenizers.__version__

        if tokenizers_version == "0.0.0.0":
            try:
                tokenizers_version = importlib_metadata.version("openvino_tokenizers") or tokenizers_version
            except importlib_metadata.PackageNotFoundError:
                pass
        message = (
            "OpenVINO and OpenVINO Tokenizers versions are not binary compatible.\n"
            f"OpenVINO version:            {_openvino_version}\n"
            f"OpenVINO Tokenizers version: {tokenizers_version}\n"
            "First 3 numbers should be the same. Update OpenVINO Tokenizers to compatible version. "
        )
        if not pip_metadata_version:
            message += (
                "For archive installation of OpenVINO try to build OpenVINO Tokenizers from source: "
                "https://github.com/openvinotoolkit/openvino_tokenizers/tree/master?tab=readme-ov-file"
                "#build-and-install-from-source"
            )
            if sys.platform == "linux":
                message += (
                    "\nThe PyPI version of OpenVINO Tokenizers is built on CentOS and may not be compatible with other "
                    "Linux distributions; rebuild OpenVINO Tokenizers from source."
                )
        else:
            message += (
                "It is recommended to use the same day builds for pre-release version. "
                "To install both OpenVINO and OpenVINO Tokenizers release version perform:\n"
            )
            if is_nightly:
                message += "pip uninstall -y openvino-nightly && "
            message += "pip install --force-reinstall openvino openvino-tokenizers\n"
            if is_nightly:
                message += (
                    "openvino-nightly package will be deprecated in the future - use pre-release drops instead. "
                )
            message += "To update both OpenVINO and OpenVINO Tokenizers to the latest pre-release version perform:\n"
            if is_nightly:
                message += "pip uninstall -y openvino-nightly && "
            message += (
                "pip install --pre -U openvino openvino-tokenizers "
                "--extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly"
            )
        logger.warning(message)
        return False

    return True


def is_nncf_available():
    return _nncf_available


def is_diffusers_available():
    return _diffusers_available


def is_open_clip_available():
    return _open_clip_available


def is_safetensors_available():
    return _safetensors_available


def is_timm_available():
    return _timm_available


def is_datasets_available():
    return _datasets_available


def is_pillow_available():
    return _pillow_available


def is_accelerate_available():
    return _accelerate_available


def is_sentence_transformers_available():
    return _sentence_transformers_available


def is_numa_available():
    return _numa_available


def is_psutil_available():
    return _psutil_available


# This function was copied from: https://github.com/huggingface/accelerate/blob/874c4967d94badd24f893064cc3bef45f57cadf7/src/accelerate/utils/versions.py#L319
def compare_versions(library_or_version: Union[str, Version], operation: str, requirement_version: str):
    """
    Compare a library version to some requirement using a given operation.

    Arguments:
        library_or_version (`str` or `packaging.version.Version`):
            A library name or a version to check.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`.
        requirement_version (`str`):
            The version to compare the library version against
    """
    if operation not in STR_OPERATION_TO_FUNC.keys():
        raise ValueError(f"`operation` must be one of {list(STR_OPERATION_TO_FUNC.keys())}, received {operation}")
    operation = STR_OPERATION_TO_FUNC[operation]
    if isinstance(library_or_version, str):
        library_or_version = parse(importlib_metadata.version(library_or_version))
    return operation(library_or_version, parse(requirement_version))


def is_transformers_version(operation: str, version: str):
    """
    Compare the current Transformers version to a given reference with an operation.
    """
    if not _transformers_available:
        return False
    return compare_versions(parse(_transformers_version), operation, version)


def is_tokenizers_version(operation: str, version: str):
    """
    Compare the current Tokenizers version to a given reference with an operation.
    """
    if not _tokenizers_available:
        return False
    return compare_versions(parse(_tokenizers_version), operation, version)


def is_optimum_version(operation: str, version: str):
    return compare_versions(parse(_optimum_version), operation, version)


def is_openvino_version(operation: str, version: str):
    """
    Compare the current OpenVINO version to a given reference with an operation.
    """
    if not _openvino_available:
        return False
    return compare_versions(parse(_openvino_version), operation, version)


def is_nncf_version(operation: str, version: str):
    """
    Compare the current NNCF version to a given reference with an operation.
    """
    if not _nncf_available:
        return False
    return compare_versions(parse(_nncf_version), operation, version)


def is_openvino_tokenizers_version(operation: str, version: str):
    if not is_openvino_available():
        return False
    if not is_openvino_tokenizers_available():
        return False
    import openvino_tokenizers

    tokenizers_version = openvino_tokenizers.__version__

    if tokenizers_version == "0.0.0.0":
        try:
            tokenizers_version = importlib_metadata.version("openvino_tokenizers") or tokenizers_version
        except importlib_metadata.PackageNotFoundError:
            pass

    tokenizers_version = tokenizers_version[: len("2025.0.0.0")]
    return compare_versions(parse(tokenizers_version), operation, version)


def is_diffusers_version(operation: str, version: str):
    """
    Compare the current diffusers version to a given reference with an operation.
    """
    if not _diffusers_available:
        return False
    return compare_versions(parse(_diffusers_version), operation, version)


def is_torch_version(operation: str, version: str):
    """
    Compare the current torch version to a given reference with an operation.
    """
    if not _torch_available:
        return False

    import torch

    return compare_versions(parse(parse(torch.__version__).base_version), operation, version)


def is_timm_version(operation: str, version: str):
    """
    Compare the current timm version to a given reference with an operation.
    """
    if not _timm_available:
        return False
    return compare_versions(parse(_timm_version), operation, version)


def is_datasets_version(operation: str, version: str):
    """
    Compare the current datasets version to a given reference with an operation.
    """
    if not _datasets_available:
        return False
    return compare_versions(parse(_datasets_version), operation, version)


def is_sentence_transformers_version(operation: str, version: str):
    """
    Compare the current sentence-transformers version to a given reference with an operation.
    """
    if not _sentence_transformers_available:
        return False
    return compare_versions(parse(_sentence_transformers_version), operation, version)


def is_huggingface_hub_version(operation: str, version: str):
    """
    Compare the current huggingface_hub version to a given reference with an operation.
    """
    if not _huggingface_hub_available:
        return False
    return compare_versions(parse(_huggingface_hub_version), operation, version)


DIFFUSERS_IMPORT_ERROR = """
{0} requires the diffusers library but it was not found in your environment. You can install it with pip:
`pip install diffusers`. Please note that you may need to restart your runtime after installation.
"""

NNCF_IMPORT_ERROR = """
{0} requires the nncf library but it was not found in your environment. You can install it with pip:
`pip install nncf`. Please note that you may need to restart your runtime after installation.
"""

OPENVINO_IMPORT_ERROR = """
{0} requires the openvino library but it was not found in your environment. You can install it with pip:
`pip install openvino`. Please note that you may need to restart your runtime after installation.
"""

DATASETS_IMPORT_ERROR = """
{0} requires the datasets library but it was not found in your environment. You can install it with pip:
`pip install datasets`. Please note that you may need to restart your runtime after installation.
"""

PILLOW_IMPORT_ERROR = """
{0} requires the pillow library but it was not found in your environment. You can install it with pip:
`pip install pillow`. Please note that you may need to restart your runtime after installation.
"""

ACCELERATE_IMPORT_ERROR = """
{0} requires the accelerate library but it was not found in your environment. You can install it with pip:
`pip install accelerate`. Please note that you may need to restart your runtime after installation.
"""

SENTENCE_TRANSFORMERS_IMPORT_ERROR = """
{0} requires the sentence-transformers library but it was not found in your environment. You can install it with pip:
`pip install sentence-transformers`. Please note that you may need to restart your runtime after installation.
"""


BACKENDS_MAPPING = OrderedDict(
    [
        ("diffusers", (is_diffusers_available, DIFFUSERS_IMPORT_ERROR)),
        ("nncf", (is_nncf_available, NNCF_IMPORT_ERROR)),
        ("openvino", (is_openvino_available, OPENVINO_IMPORT_ERROR)),
        ("accelerate", (is_accelerate_available, ACCELERATE_IMPORT_ERROR)),
        ("sentence_transformers", (is_sentence_transformers_available, SENTENCE_TRANSFORMERS_IMPORT_ERROR)),
    ]
)


def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError("".join(failed))


# Copied from: https://github.com/huggingface/transformers/blob/v4.26.0/src/transformers/utils/import_utils.py#L1041
class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """

    def __getattr__(cls, key):
        if key.startswith("_"):
            return super().__getattr__(cls, key)
        requires_backends(cls, cls._backends)
