#!/bin/bash

set -x

set -eo pipefail

if [ x"$JUPYTER_MASTER_FILES" != x"" ]; then
    if [ x"$JUPYTER_WORKSPACE_NAME" != x"" ]; then
        JUPYTER_WORKSPACE_PATH=/opt/app-root/src/$JUPYTER_WORKSPACE_NAME
        setup-volume.sh $JUPYTER_MASTER_FILES $JUPYTER_WORKSPACE_PATH
    fi
fi

JUPYTER_PROGRAM_ARGS="$JUPYTER_PROGRAM_ARGS --config=/opt/app-root/etc/jupyter_kernel_gateway_config.py"

exec /opt/app-root/bin/start.sh jupyter kernelgateway $JUPYTER_PROGRAM_ARGS "$@"
