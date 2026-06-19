# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

NNCF_PACKAGE_ROOT_DIR = Path(__file__).resolve().parent
HW_CONFIG_RELATIVE_DIR = "common/hardware/configs"

NNCF_CACHE_PATH = Path.home() / Path(".cache/nncf/")
CACHE_MODELS_PATH = NNCF_CACHE_PATH / Path("models")

# Environment variables below, if set, mark the execution environment
# so that certain actions within NNCF proper, such as telemetry event collection or
# debug dumps, are performed or not performed
NNCF_CI_ENV_VAR_NAME = "NNCF_CI"  # Must be set in CI environments
NNCF_DEV_ENV_VAR_NAME = "NNCF_DEV"  # Must be set in environments of the NNCF dev team machines

# This is a special input key used by OpenVINO backend to control resetting of internal model state
# between model inferences. This key can be added to a dataset sample input dictionary with either
# `True` or `False` value. With `True` value, the model state will be reset before inference on the corresponding
# sample, and with `False` the state will not be reset.
NNCF_DATASET_RESET_STATE_KEY = "nncf_reset_state"
