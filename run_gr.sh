#!/bin/bash

source .venv/bin/activate

arch=$(uname -m)
if [[ "$arch" == "aarch64" ]]; then
    export LD_LIBRARY_PATH="/lib/aarch64-linux-gnu:/opt/sophon/libsophon-current/lib:$LD_LIBRARY_PATH"
elif [[ "$arch" == "x86_64" ]]; then
    export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/opt/sophon/libsophon-current/lib:$LD_LIBRARY_PATH"
else
    echo "Unsupported architecture: $arch"
fi

export PYTHONPATH=/opt/sophon/sophon-opencv-latest/opencv-python/:$PYTHONPATH

python3 gr.py


