#!/bin/bash

set -x

set -eo pipefail

if [[ ! -z "${JUPYTER_ENABLE_KERNELGATEWAY}" ]]; then
    exec /opt/app-root/bin/start-kernelgateway.sh "$@"
fi

if [ x"$JUPYTER_MASTER_FILES" != x"" ]; then
    if [ x"$JUPYTER_WORKSPACE_NAME" != x"" ]; then
        JUPYTER_WORKSPACE_PATH=/opt/app-root/src/$JUPYTER_WORKSPACE_NAME
        setup-volume.sh $JUPYTER_MASTER_FILES $JUPYTER_WORKSPACE_PATH
    fi
fi

JUPYTER_PROGRAM_ARGS="$JUPYTER_PROGRAM_ARGS $NOTEBOOK_ARGS"

JUPYTER_NOTEBOOK_INTERFACE=${JUPYTER_NOTEBOOK_INTERFACE:-classic}

if [ x"$JUPYTER_ENABLE_LAB" = x"" ]; then
    if [ x"$JUPYTER_NOTEBOOK_INTERFACE" = x"lab" ]; then
        JUPYTER_ENABLE_LAB=true
    fi
fi

JUPYTER_ENABLE_LAB=`echo "$JUPYTER_ENABLE_LAB" | tr '[A-Z]' '[a-z]'`

if [[ "$JUPYTER_ENABLE_LAB" =~ ^(true|yes|y|1)$ ]]; then
    JUPYTER_PROGRAM_ARGS="$JUPYTER_PROGRAM_ARGS --NotebookApp.default_url=/lab"
else
    if [ x"$JUPYTER_WORKSPACE_NAME" != x"" ]; then
        JUPYTER_PROGRAM_ARGS="$JUPYTER_PROGRAM_ARGS --NotebookApp.default_url=/tree/$JUPYTER_WORKSPACE_NAME"
    fi
fi

if [[ "$JUPYTER_ENABLE_LAB" =~ ^(true|yes|y|1)$ ]]; then
    JUPYTER_PROGRAM_ARGS="$JUPYTER_PROGRAM_ARGS --config=/opt/app-root/etc/jupyter_server_config.py"
else
    JUPYTER_PROGRAM_ARGS="$JUPYTER_PROGRAM_ARGS --config=/opt/app-root/etc/jupyter_notebook_config.py"
fi

if [[ "$JUPYTER_PROGRAM_ARGS $@" != *"--ip="* ]]; then
    JUPYTER_PROGRAM_ARGS="--ip=0.0.0.0 $JUPYTER_PROGRAM_ARGS"
fi

if [[ ! -z "${JUPYTERHUB_API_TOKEN}" ]]; then
    if [[ "$JUPYTER_ENABLE_LAB" =~ ^(true|yes|y|1)$ ]]; then
        JUPYTER_PROGRAM="jupyter labhub"
    else
        JUPYTER_PROGRAM="jupyterhub-singleuser"
    fi
else
    if [[ "$JUPYTER_ENABLE_LAB" =~ ^(true|yes|y|1)$ ]]; then
        JUPYTER_PROGRAM="jupyter lab"
    else
        JUPYTER_PROGRAM="jupyter notebook"
    fi
fi

. /opt/app-root/bin/setup-environ.sh

if [ -f /opt/app-root/src/.jupyter/jupyter_notebook_config.sh ]; then
    . /opt/app-root/src/.jupyter/jupyter_notebook_config.sh
fi

exec /opt/app-root/bin/start.sh $JUPYTER_PROGRAM $JUPYTER_PROGRAM_ARGS "$@"
