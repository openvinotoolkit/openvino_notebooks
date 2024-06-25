#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
FROM nvcr.io/nvidia/tritonserver:23.07-py3

RUN python3 -m pip install --upgrade pip && \
    pip install \
        openvino==2023.1.0.dev20230728 \
        git+https://github.com/huggingface/optimum-intel.git \
        onnx onnxruntime einops
