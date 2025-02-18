FROM quay.io/opendatahub/workbench-images:runtime-minimal-ubi9-python-3.9

LABEL name="odh-notebook-jupyter-intel-openvino-ubi9-python-3.9" \
    maintainer="helena.kloosterman@intel.com" \
    vendor="Intel Corporation" \
    release="2024.2" \
    summary="Jupyter Intel® OpenVINO notebook image for ODH notebooks." \
    description="Jupyter Intel® OpenVINO notebook image with base Python 3.9 builder image based on UBI9 for ODH notebooks" \
    io.k8s.display-name="Jupyter Intel® OpenVINO notebook image for ODH notebooks." \
    io.k8s.description="Jupyter Intel® OpenVINO notebook image with base Python 3.9 builder image based on UBI9 for ODH notebooks" \
    io.openshift.build.commit.ref="main"

USER 0
WORKDIR /opt/app-root/src

RUN . /etc/os-release && \
    #TODO: Remove explicit declaration of VERSION_ID once available on version 9.4
    VERSION_ID=9.4 && \
    dnf install -y 'dnf-command(config-manager)' && \
    dnf config-manager --add-repo \
        https://repositories.intel.com/gpu/rhel/${VERSION_ID}/unified/intel-gpu-${VERSION_ID}.repo  && \
    dnf install -y \
        intel-opencl \
        level-zero intel-level-zero-gpu level-zero-devel && \
    rpm -ivh https://dl.fedoraproject.org/pub/epel/9/Everything/x86_64/Packages/c/clinfo-3.0.21.02.21-4.el9.x86_64.rpm  \
    https://mirror.stream.centos.org/9-stream/AppStream/x86_64/os/Packages/ocl-icd-2.2.13-4.el9.x86_64.rpm && \
    yum install -y libsndfile && \
    dnf clean all -y && \
    rm -rf /var/cache/dnf/*

USER 1001

WORKDIR /opt/app-root/bin

# Install Python packages and Jupyterlab extensions from Pipfile.lock
COPY .docker/Pipfile.lock Pipfile.lock
RUN echo "Installing softwares and packages" && \
    micropipenv install && \
    rm -f ./Pipfile.lock && \
    # Disable announcement plugin of jupyterlab \
    jupyter labextension disable "@jupyterlab/apputils-extension:announcements" && \
    chmod -R g+w /opt/app-root/lib/python3.9/site-packages && \
    fix-permissions /opt/app-root -P

COPY --chown=1001:0 .docker/start-notebook.sh /opt/app-root/bin
COPY --chown=1001:0 .docker/builder /opt/app-root/builder
COPY --chown=1001:0 .docker/utils /opt/app-root/bin/utils

COPY --chown=1001:0 .docker/tests/test /tmp/scripts/test
COPY --chown=1001:0 .docker/tests/test_precommit /tmp/scripts
COPY --chown=1001:0 .ci/patch_notebooks.py /tmp/scripts
COPY --chown=1001:0 .ci/validate_notebooks.py /tmp/scripts
COPY --chown=1001:0 .ci/validation_config.py /tmp/scripts
COPY --chown=1001:0 .ci/ignore_treon_docker.txt /tmp/scripts
# workaround for coping file if it does not exists
COPY --chown=1001:0 .ci/test_notebooks.* /tmp/scripts


WORKDIR /opt/app-root/src

ENV JUPYTER_PRELOAD_REPOS="https://github.com/openvinotoolkit/openvino_notebooks"
ENV REPO_BRANCH="latest"

ENTRYPOINT ["bash", "-c", "/opt/app-root/builder/run"]
