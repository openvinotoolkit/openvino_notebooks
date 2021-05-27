# S2I Minimal Notebook

Minimal Thoth S2I notebook builder

This repository is a Fork of Graham Dumpleton: [jupyter-on-openshift/jupyter-notebooks](https://github.com/jupyter-on-openshift/jupyter-notebooks).

This repository contains Source-to-Image (S2I) build process to create a Minimal Jupyter Notebooks on OpenShift. The image can be built in OpenShift, separately using the `s2i` tool, or using a `docker build`. The same image, can also be used as an S2I builder to create customised Jupyter notebook images with additional Python packages installed, or notebook files preloaded.

## Importing the Minimal Notebook

A pre-built version of the minimal notebook which is based on Thoth Ubi8 Python36, can be found at on quay.io at:

- <https://quay.io/repository/thoth-station/s2i-minimal-notebook> [![Docker Repository on Quay](https://quay.io/repository/thoth-station/s2i-minimal-notebook/status "Docker Repository on Quay")](https://quay.io/repository/thoth-station/s2i-minimal-notebook)

This image could be imported into an OpenShift cluster using OpenShift ImageStream:

```yaml
apiVersion: image.openshift.io/v1
kind: ImageStream
metadata:
  # (Below label is needed for Opendatahub.io/JupyterHub)
  # labels:
  #   opendatahub.io/notebook-image: "true"
  name: s2i-minimal-notebook
spec:
  lookupPolicy:
    local: true
  tags:
  - name: latest
    from:
      kind: DockerImage
      name: quay.io/thoth-station/s2i-minimal-notebook:latest
```

## Building the Minimal Notebook

Instead of using the pre-built version of the minimal notebook, you can build the minimal notebook from source code.

With [Thoth](https://thoth-station.ninja/) advise

```bash
s2i build . quay.io/thoth-station/s2i-thoth-ubi8-py36:latest \
--env ENABLE_PIPENV=1 \
--env THOTH_ADVISE=1 \
--env THOTH_DRY_RUN=0 \
--env THOTH_PROVENANCE_CHECK=1 \
s2i-minimal-notebook
```

Without [Thoth](https://thoth-station.ninja/) advise

```bash
s2i build . quay.io/thoth-station/s2i-thoth-ubi8-py36:latest \
--env ENABLE_PIPENV=1 \
--env THOTH_ADVISE=0 \
--env THOTH_ERROR_FALLBACK=1 \
--env THOTH_DRY_RUN=1 \
--env THOTH_PROVENANCE_CHECK=0 \
s2i-minimal-notebook
```
