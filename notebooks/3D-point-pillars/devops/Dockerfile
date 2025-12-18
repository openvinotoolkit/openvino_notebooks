ARG PYTHON_VERSION=3.10
ARG UBUNTU_VERSION=24.04

ARG http_proxy
ARG https_proxy
ARG no_proxy

FROM ubuntu:${UBUNTU_VERSION} AS base-selected

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies
RUN apt update && \
    apt install -y --no-install-recommends \
    software-properties-common \
    cmake \
    build-essential \
    libx11-6 \
    libgl1

# Install Python and pip
ARG PYTHON_VERSION
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Upgrade pip, setuptools, wheel, blinker
RUN python -m pip install --no-cache-dir --upgrade --ignore-installed pip setuptools wheel blinker

# intel GPU
RUN cd /usr/lib/python3/dist-packages/ && \
    ln -s apt_pkg.cpython-312-x86_64-linux-gnu.so apt_pkg.so

RUN add-apt-repository -y ppa:kobuk-team/intel-graphics || true
RUN apt update && \
    apt install -y --no-install-recommends \
    libze-intel-gpu1 \
    intel-opencl-icd




# Set working directory
WORKDIR /workspace

# Install Python packages
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Building torch extensions...
COPY pointpillars/ ./pointpillars/
COPY setup.py .
RUN CPU_BUILD=1 python setup.py build_ext --inplace

# Building OpenVINO extensions...
COPY ov_extensions ./ov_extensions
RUN cd ov_extensions && rm -rf build && bash build.sh && cd ..

# Exporting OpenVINO model
COPY export_ov_e2e.py .
COPY pretrained ./pretrained
# python export_ov_e2e.py --checkpoint pretrained/epoch_160.pth --output pretrained/pointpillars_ov

# Test
COPY e2eOVInference.py .
COPY test-e2eOV.py .
# python test-e2eOV.py --device CPU
# python test-e2eOV.py --device GPU

# Evaluation Kitti
COPY evaluate-e2eOV.py .
# python evaluate-e2eOV.py --data_root /Datasets --nsamples 10
