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
  THOTH_PROVENANCE_CHECK="0" \
  JUPYTER_PRELOAD_REPOS="https://github.com/openvinotoolkit/openvino_notebooks"

USER root

# Upgrade NodeJS > 12.0
# Install dos2unix for line end conversion on Windows
RUN curl -sL https://rpm.nodesource.com/setup_14.x | bash -  && \
  yum remove -y nodejs && \
  yum install -y nodejs mesa-libGL dos2unix && \
  yum -y update-minimal --security --sec-severity=Important --sec-severity=Critical --sec-severity=Moderate

# install iGPU dependencies
ARG GPU=False
RUN if [ "$GPU" = "True" ]; then \
    set -e ; \
    set -x ; \
    mkdir /tmp/gpu_deps ; \
    curl -L --output /tmp/gpu_deps/intel-opencl-20.35.17767-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/20.35.17767/centos-7/intel-opencl-20.35.17767-1.el7.x86_64.rpm/download ; \
    curl -L --output /tmp/gpu_deps/level-zero-1.0.0-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/20.35.17767/centos-7/level-zero-1.0.0-1.el7.x86_64.rpm/download ; \
    curl -L --output /tmp/gpu_deps/level-zero-devel-1.0.0-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/20.35.17767/centos-7/level-zero-devel-1.0.0-1.el7.x86_64.rpm/download ; \
    curl -L --output /tmp/gpu_deps/intel-igc-opencl-1.0.4756-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/20.35.17767/centos-7/intel-igc-opencl-1.0.4756-1.el7.x86_64.rpm/download ; \
    curl -L --output /tmp/gpu_deps/intel-igc-opencl-devel-1.0.4756-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/20.35.17767/centos-7/intel-igc-opencl-devel-1.0.4756-1.el7.x86_64.rpm/download ; \
    curl -L --output /tmp/gpu_deps/intel-igc-core-1.0.4756-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/20.35.17767/centos-7/intel-igc-core-1.0.4756-1.el7.x86_64.rpm/download ; \
    curl -L --output /tmp/gpu_deps/intel-gmmlib-20.2.4-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/20.35.17767/centos-7/intel-gmmlib-20.2.4-1.el7.x86_64.rpm/download ; \
    curl -L --output /tmp/gpu_deps/intel-gmmlib-devel-20.2.4-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/20.35.17767/centos-7/intel-gmmlib-devel-20.2.4-1.el7.x86_64.rpm/download ; \
    rpm -ivh http://mirror.centos.org/centos/8-stream/BaseOS/x86_64/os/Packages/numactl-libs-2.0.12-11.el8.x86_64.rpm; \
    rpm -ivh http://mirror.centos.org/centos/8/BaseOS/x86_64/os/Packages/numactl-2.0.12-11.el8.x86_64.rpm; \
    rpm -ivh http://mirror.centos.org/centos/8/AppStream/x86_64/os/Packages/ocl-icd-2.2.12-1.el8.x86_64.rpm; \
    dnf install shadow-utils; \
    cd /tmp/gpu_deps && rpm -iv *.rpm && rm -Rf /tmp/gpu_deps ; \
  else \
    echo "Skipping GPU dependencies" ; \
  fi

# Copying in override assemble/run scripts
COPY .docker/.s2i/bin /tmp/scripts
# Copying in source code
COPY .docker /tmp/src
COPY .ci/patch_notebooks.py /tmp/scripts

# Git on Windows may convert line endings. Run dos2unix to enable
# building the image when the scripts have CRLF line endings.
RUN dos2unix /tmp/scripts/*
RUN dos2unix /tmp/src/builder/*

# Change file ownership to the assemble user. Builder image must support chown command.
RUN chown -R 1001:0 /tmp/scripts /tmp/src
USER 1001
RUN /tmp/scripts/assemble
RUN pip check
USER root
RUN dos2unix /opt/app-root/bin/*sh
RUN yum remove -y dos2unix
RUN chown -R 1001:0 .
USER 1001
# RUN jupyter lab build
CMD /tmp/scripts/run
