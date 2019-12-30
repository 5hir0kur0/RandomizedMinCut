#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

NODE_COUNTS=(10 100 1000 2000 3000)

function maybe_compile {
    if [ GraphGenerator.java -nt GraphGenerator.class ]; then
        javac GraphGenerator.java
    fi
}

function main {
    maybe_compile
    density=${1:?expected density}
    bench_dir="bench_$density"
    mkdir "$bench_dir"
    for count in "${NODE_COUNTS[@]}"; do
        java GraphGenerator \
            "$count" "$density" 1 "$bench_dir/graph${count}_${density}.txt"
    done
}

main "$@"
