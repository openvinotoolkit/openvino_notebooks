#!/bin/bash

set -x

set -eo pipefail

JUPYTER_ENABLE_LAB=true
export JUPYTER_ENABLE_LAB

exec /opt/app-root/bin/start-notebook.sh "$@"
