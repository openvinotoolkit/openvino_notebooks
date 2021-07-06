#!/bin/bash

set -x

if [ $# -eq 0 ]; then
    echo "Executing the command: bash"
    exec bash
else
    echo "Executing the command: $@"

    if [[ -z "${JUPYTER_ENABLE_SUPERVISORD}" ]]; then
        trap 'kill -TERM $PID' TERM INT

        "$@" &

        PID=$!
        wait $PID
        trap - TERM INT
        wait $PID
        STATUS=$?
        exit $STATUS
    else
        exec "$@"
    fi
fi
