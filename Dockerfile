FROM quay.io/thoth-station/s2i-thoth-ubi8-py38:v0.29.0

LABEL name="OpenVINO(TM) Notebooks" \
  maintainer="helena.kloosterman@intel.com" \
  vendor="Intel Corporation" \
  version="0.2.0" \
  release="2021.4" \
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
RUN dnf --disableplugin=subscription-manager remove -y nodejs && \
  dnf --disableplugin=subscription-manager module -y reset nodejs && \
  dnf --disableplugin=subscription-manager module -y enable nodejs:20 && \
  dnf --disableplugin=subscription-manager install -y nodejs mesa-libGL dos2unix libsndfile && \
  dnf --disableplugin=subscription-manager -y update-minimal --security --sec-severity=Important --sec-severity=Critical --sec-severity=Moderate

# GPU drivers
RUN dnf --disableplugin=subscription-manager install -y 'dnf-command(config-manager)' && \
    dnf --disableplugin=subscription-manager config-manager --add-repo  https://repositories.intel.com/graphics/rhel/8.6/intel-graphics.repo

RUN rpm -ivh https://vault.centos.org/centos/8/AppStream/x86_64/os/Packages/mesa-filesystem-21.1.5-1.el8.x86_64.rpm && \
    dnf --disableplugin=subscription-manager install --refresh -y \
    intel-opencl intel-media intel-mediasdk libmfxgen1 libvpl2 \
    level-zero intel-level-zero-gpu \
    intel-metrics-library intel-igc-core intel-igc-cm \
    libva libva-utils  intel-gmmlib && \
    rpm -ivh http://mirror.centos.org/centos/8-stream/AppStream/x86_64/os/Packages/ocl-icd-2.2.12-1.el8.x86_64.rpm && \
    rpm -ivh https://dl.fedoraproject.org/pub/epel/8/Everything/x86_64/Packages/c/clinfo-3.0.21.02.21-4.el8.x86_64.rpm

# Copying in override assemble/run scripts
COPY .docker/.s2i/bin /tmp/scripts
# Copying in source code
COPY .docker /tmp/src
COPY .ci/patch_notebooks.py /tmp/scripts
COPY .ci/validate_notebooks.py /tmp/scripts
COPY .ci/ignore_treon_docker.txt /tmp/scripts
# workaround for coping file if it does not exists
COPY .ci/test_notebooks.tx[t] /tmp/scripts

# Git on Windows may convert line endings. Run dos2unix to enable
# building the image when the scripts have CRLF line endings.
RUN dos2unix /tmp/scripts/*
RUN dos2unix /tmp/src/builder/*

# Change file ownership to the assemble user. Builder image must support chown command.
RUN chown -R 1001:0 /tmp/scripts /tmp/src
USER 1001
RUN mkdir /opt/app-root/notebooks
COPY notebooks/ /opt/app-root/notebooks
RUN /tmp/scripts/assemble
RUN pip check
USER root
RUN dos2unix /opt/app-root/bin/*sh
RUN yum remove -y dos2unix
RUN chown -R 1001:0 .
RUN chown -R 1001:0 /opt/app-root/notebooks
USER 1001
# RUN jupyter lab build
CMD /tmp/scripts/run
