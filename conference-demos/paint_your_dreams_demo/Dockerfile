FROM quay.io/thoth-station/s2i-thoth-ubi8-py38:v0.29.0

LABEL name="OpenVINO(TM) Notebooks" \
  maintainer="patryk.lange@intel.com" \
  vendor="Intel Corporation" \
  version="0.2.1" \
  release="2023.3" \
  summary="OpenVINO(TM) Developer Tools and Jupyter Notebooks" \
  description="OpenVINO(TM) Notebooks Container"

ENV JUPYTER_ENABLE_LAB="true" \
  ENABLE_MICROPIPENV="1" \
  UPGRADE_PIP_TO_LATEST="1" \
  WEB_CONCURRENCY="1" \
  THOTH_ADVISE="0" \
  THOTH_ERROR_FALLBACK="1" \
  THOTH_DRY_RUN="1" \
  THAMOS_DEBUG="0" \
  THAMOS_VERBOSE="1" \
  THOTH_PROVENANCE_CHECK="0"

USER root

# Upgrade NodeJS > 12.0
# Install dos2unix for line end conversion on Windows
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -sL https://rpm.nodesource.com/setup_14.x | bash -  && \
  yum remove -y nodejs && \
  yum install -y nodejs-14.18.1 mesa-libGL dos2unix libsndfile --setopt=tsflags=nocrypto --nogpgcheck && \
  yum -y update-minimal --security --sec-severity=Important --sec-severity=Critical --sec-severity=Moderate

# GPU drivers
RUN dnf install -y 'dnf-command(config-manager)' && \
    dnf config-manager --add-repo  https://repositories.intel.com/graphics/rhel/8.5/intel-graphics.repo

RUN rpm -ivh https://vault.centos.org/centos/8/AppStream/x86_64/os/Packages/mesa-filesystem-21.1.5-1.el8.x86_64.rpm && \
    dnf install --refresh -y \
    intel-opencl-22.28.23726.1-i419.el8.x86_64 intel-media intel-mediasdk libmfxgen1 libvpl2 \
    level-zero intel-level-zero-gpu \
    intel-metrics-library intel-igc-core intel-igc-cm \
    libva libva-utils  intel-gmmlib && \
    rpm -ivh http://mirror.centos.org/centos/8-stream/AppStream/x86_64/os/Packages/ocl-icd-2.2.12-1.el8.x86_64.rpm && \
    rpm -ivh https://dl.fedoraproject.org/pub/epel/8/Everything/x86_64/Packages/c/clinfo-3.0.21.02.21-4.el8.x86_64.rpm

# Copying in override assemble/run scripts
COPY .docker/.s2i/bin /tmp/scripts
# Copying in source code
COPY .docker /tmp/src

# Git on Windows may convert line endings. Run dos2unix to enable
# building the image when the scripts have CRLF line endings.
RUN dos2unix /tmp/scripts/*
RUN dos2unix /tmp/src/builder/*

# Change file ownership to the assemble user. Builder image must support chown command.
RUN chown -R 1001:0 /tmp/scripts /tmp/src
USER 1001
RUN mkdir /opt/app-root/notebooks
COPY /conference-demos/paint-your-dreams-demo/263-lcm-image-generation-paint-your-dreams.ipynb /opt/app-root/notebooks/
RUN chmod +x /tmp/scripts/assemble
RUN /tmp/scripts/assemble
RUN pip check
USER root
RUN dos2unix /opt/app-root/bin/*sh
RUN yum remove -y dos2unix
RUN chown -R 1001:0 .
RUN chown -R 1001:0 /opt/app-root/notebooks
USER 1001

# Install additional packages
RUN pip install "torch==2.3.1" --index-url https://download.pytorch.org/whl/cpu
RUN pip install "openvino==2024.1.0" "transformers==4.41.2" "diffusers==0.28.2" "pillow==10.3.0" "gradio==4.33.0" "nncf==2.10.0" "datasets==2.19.2" --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install "accelerate==0.30.1"

# RUN jupyter lab build
CMD ["/tmp/scripts/run"]
