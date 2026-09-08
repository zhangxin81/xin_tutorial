#!/usr/bin/env bash
set -euo pipefail

# Use a private directory so several users do not collide accidentally.
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps-${USER}}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log-${USER}}"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

case "${1:-status}" in
  start)
    # Optional: export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 before start/run.
    # This is a limit, not a reservation of dedicated SMs.
    nvidia-cuda-mps-control -d
    echo "MPS started"
    echo "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
    echo "CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY"
    ;;
  stop)
    echo quit | nvidia-cuda-mps-control
    echo "MPS stopped"
    ;;
  status)
    echo get_server_list | nvidia-cuda-mps-control || true
    echo "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
    echo "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-unset}"
    ;;
  *)
    echo "Usage: source $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
