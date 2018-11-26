#!/bin/bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS=...
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

PORT=$1

source activate segmentation-transfer
ssh -N -R $PORT:localhost:$PORT elisa1 &
python -m visdom.server -port $PORT
