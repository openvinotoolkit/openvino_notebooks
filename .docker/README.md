# Building the image

```bash
docker build . -t openvino_notebooks
```

# Using the image

The command below starts the jupyterlab with default parameters on port 8888
```bash
docker run -d -p 8888:8888  openvino_notebooks
```

You can also add extra parameters and mount storage as needed. For example:
```bash
docker run -it -v $(pwd):/workspace -u $(id -u) -p 8080:8080 -e NOTEBOOK_PORT=8080 -e NOTEBOOK_ROOT_DIR="/workspace" -e NOTEBOOK_ARGS="--NotebookApp.token=''" openvino_notebooks
```

# Testing the image

```bash
docker run -it  --entrypoint /tmp/scripts/test -v ${PWD}:/opt/app-root/openvino_notebooks openvino_notebooks
```

The PR workflow uses `.docker/tests/test_precommit`. It temporarily installs a Python startup hook in the test container that deduplicates resolved import paths only for `optimum-cli`, avoiding duplicate command registration through UBI's `lib`/`lib64` aliases. The hook is inherited by notebook virtual environments; it does not patch Optimum or shared notebook helpers and is not installed when building or normally starting the image. Remove this workaround once Optimum CLI handles aliased namespace paths.

# Updating Pipfile.lock

```bash
 docker run -v $(pwd)/.docker/Pipfile:/opt/app-root/bin/Pipfile --entrypoint bash openvino_notebooks:latest -c 'cd ../bin ; pip install -qq pipenv ; pipenv install  >/dev/null 2>&1 ; cat Pipfile.lock' > .docker/Pipfile.lock
```
