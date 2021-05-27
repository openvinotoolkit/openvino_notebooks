#!/bin/bash

set -eo pipefail

set -x

WEBDAV_PREFIX=${WEBDAV_PREFIX:-${JUPYTERHUB_SERVICE_PREFIX%/}}
export WEBDAV_PREFIX

WEBDAV_PORT=${WEBDAV_PORT:-8081}

ARGS=""

ARGS="$ARGS --log-to-terminal"
ARGS="$ARGS --access-log"
ARGS="$ARGS --port $WEBDAV_PORT"
ARGS="$ARGS --application-type static"
ARGS="$ARGS --include /opt/app-root/etc/httpd-webdav.conf"

exec mod_wsgi-express start-server $ARGS
