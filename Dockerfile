FROM quay.io/thoth-station/s2i-thoth-ubi8-py38:v0.29.0

LABEL name="OpenVINO(TM) Notebooks" \
  maintainer="helena.kloosterman@intel.com" \
  vendor="Intel Corporation" \
  version="0.1.0" \
  release="2021.3" \
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
RUN curl -sL https://rpm.nodesource.com/setup_14.x | bash -  && \
  yum remove -y nodejs && \
  yum install -y nodejs mesa-libGL && \
  yum -y update-minimal --security --sec-severity=Important --sec-severity=Critical --sec-severity=Moderate

# Copying in override assemble/run scripts
COPY .openshift/.s2i/bin /tmp/scripts
# Copying in source code
COPY .openshift /tmp/src
# Change file ownership to the assemble user. Builder image must support chown command.
RUN chown -R 1001:0 /tmp/scripts /tmp/src
USER 1001
RUN /tmp/scripts/assemble

# These manual pip installs will be removed before final release and added to the Piplock file

COPY notebooks .
COPY openvino-2021.4.0-3810-cp38-cp38-manylinux2014_x86_64.whl .
COPY openvino_dev-2021.4.0-3810-py3-none-any.whl .


RUN pip install openvino-2021.4.0-3810-cp38-cp38-manylinux2014_x86_64.whl openvino_dev-2021.4.0-3810-py3-none-any.whl tensorflow-serving-api nbval --use-deprecated=legacy-resolver

USER root
RUN chown -R 1001:0 .
USER 1001
# RUN jupyter lab build
CMD /tmp/scripts/run
