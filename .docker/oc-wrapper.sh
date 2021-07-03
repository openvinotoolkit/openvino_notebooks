#!/bin/bash

case $OC_VERSION in
    4.*)
        OC_VERSION=4
        ;;
    *)
        OC_VERSION=3.11
        ;;
esac

exec /opt/app-root/bin/oc-$OC_VERSION "$@"
