#!/bin/bash

set -x

set -eo pipefail

export $(cgroup-limits)

DASK_SCHEDULER_ADDRESS=${DASK_SCHEDULER_ADDRESS:-127.0.0.1:8786}

exec /opt/app-root/bin/start.sh dask-worker $DASK_SCHEDULER_ADDRESS \
    --memory-limit $MEMORY_LIMIT_IN_BYTES $DASK_WORKER_ARGS "$@"
