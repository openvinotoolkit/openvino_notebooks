FROM quay.io/thoth-station/s2i-thoth-ubi8-py38:v0.26.0
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
  yum install -y nodejs mesa-libGL

# Copying in override assemble/run scripts
COPY .s2i/bin /tmp/scripts
# Copying in source code
COPY . /tmp/src
# Change file ownership to the assemble user. Builder image must support chown command.
RUN chown -R 1001:0 /tmp/scripts /tmp/src
USER 1001
RUN /tmp/scripts/assemble
RUN pip install openvino-dev
RUN pip install --upgrade jupyterlab ipympl
RUN pip install git+https://github.com/maartenbreddels/ipyvolume.git
RUN git clone https://github.com/openvinotoolkit/openvino_notebooks.git
COPY testing_notebooks openvino_notebooks/testing_notebooks
USER root
RUN chown -R 1001:0 openvino_notebooks
USER 1001
RUN jupyter lab build
CMD /tmp/scripts/run

