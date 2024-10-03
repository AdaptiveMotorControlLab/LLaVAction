#!/bin/bash

# Export environment variables
export CUDA_VISIBLE_DEVICES="0"
# export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libffi.so.7"

# Run the Python script
python3 docs/LLaVA_OneVision_Tutorials.py > demo7b.out 2>&1