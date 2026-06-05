#!/bin/bash
# Plan B Evaluation - Minimal Script

cd /home/henrylin0822/coding/SQL/Database_final

# Set clean environment
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
export LD_LIBRARY_PATH=/usr/lib/nvidia-cuda-toolkit/lib64:$LD_LIBRARY_PATH

# Run Plan B evaluation
/usr/bin/python3 scripts/run_plan_b_evaluation.py --config configs/plan_b_config.yaml "${@}"