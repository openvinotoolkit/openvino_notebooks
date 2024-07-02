FROM quay.io/opendatahub/workbench-images:intel-runtime-ml-ubi9-python-3.9

LABEL name="odh-notebook-jupyter-intel-ml-ubi9-python-3.9" \
    maintainer="helena.kloosterman@intel.com" \
    vendor="Intel Corporation" \
    release="2024.2" \
    summary="Jupyter Intel® OpenVINO notebook image for ODH notebooks." \
    description="Jupyter Intel® OpenVINO notebook image with base Python 3.9 builder image based on UBI9 for ODH notebooks" \
    io.k8s.display-name="Jupyter Intel® OpenVINO notebook image for ODH notebooks." \
    io.k8s.description="Jupyter Intel® OpenVINO notebook image with base Python 3.9 builder image based on UBI9 for ODH notebooks" \
    io.openshift.build.commit.ref="main"

WORKDIR /opt/app-root/bin

# Install Python packages and Jupyterlab extensions from Pipfile.lock
COPY .docker/Pipfile.lock Pipfile.lock

COPY --chown=1001:0 .docker/.patch_sklearn.py /opt/app-root/bin/.patch_sklearn.py
ENV PYTHONSTARTUP="/opt/app-root/bin/.patch_sklearn.py"

RUN echo "Installing softwares and packages" && \
    micropipenv install && \
    rm -f ./Pipfile.lock && \
    # Disable announcement plugin of jupyterlab \
    jupyter labextension disable "@jupyterlab/apputils-extension:announcements" && \
    chmod -R g+w /opt/app-root/lib/python3.9/site-packages && \
    fix-permissions /opt/app-root -P

#Replacing kernel manually with oneapi variable setting script
COPY --chown=1001:0 .docker/start-notebook.sh /opt/app-root/bin
COPY --chown=1001:0 .docker/builder /opt/app-root/builder
COPY --chown=1001:0 .docker/utils /opt/app-root/bin/utils

WORKDIR /opt/app-root/src

ENV JUPYTER_PRELOAD_REPOS="https://github.com/openvinotoolkit/openvino_notebooks"
ENV REPO_BRANCH="main"

ENTRYPOINT ["bash", "-c", "/opt/app-root/builder/run"]