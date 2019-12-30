#!/bin/sh

if [ -n "$DEBUG" ]; then
    echo 'cargo run --release -- ' "$@"
    cargo run --release -- "$@"
else
    echo 'cargo run --release -- ' "$@" '2>/dev/null'
    cargo run --release -- "$@" 2>/dev/null
fi
