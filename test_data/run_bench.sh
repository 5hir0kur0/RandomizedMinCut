#!/bin/bash

set -eo pipefail

cd "$(dirname "$0")"

function main {
    bench=${1:?expected benchmark name}
    benchfile="${bench}_results.txt"
    echo '::::BENCHMARK START::::' >> "$benchfile"
    readarray -t files < <(find "$bench" -type f | sort -V)
    for mark in "${files[@]}"; do
        echo "running benchmark $mark"
        if [ -n "${2}" ]; then
            ./run.sh "$2" "$mark" 1>> "$benchfile"
        else
            ./run.sh "$mark" 1>> "$benchfile"
        fi
    done
}

main "$@"
