#!/bin/bash

printf "READY\n";

while read line; do
    echo "Processing Event: $line" >&2;
    supervisorctl shutdown
done < /dev/stdin
