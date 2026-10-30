#!/usr/bin/env bash
# SLAI: allocated GPUs; independent stagewise experiment.
set -eo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" phase "$@"
