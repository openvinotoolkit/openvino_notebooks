#!/bin/bash

set -x

set -eo pipefail

exec /opt/app-root/bin/start.sh dask-scheduler $DASK_SCHEDULER_ARGS "$@"
