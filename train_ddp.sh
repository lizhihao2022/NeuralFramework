#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:-}"
      shift 2
      ;;
    --config=*)
      CONFIG_PATH="${1#*=}"
      shift 1
      ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 --config /path/to/config.yaml" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CONFIG_PATH" ]]; then
  echo "Missing --config" >&2
  echo "Usage: $0 --config /path/to/config.yaml" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py --config "$CONFIG_PATH"
