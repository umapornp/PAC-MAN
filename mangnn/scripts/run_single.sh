#!/usr/bin/env bash

NUM_TRAINERS=2
CONFIG=mangnn/config/config.yaml

torchrun \
--standalone \
--nnodes=1 \
--nproc_per_node=${1-$NUM_TRAINERS} \
mangnn/run.py \
--cfg=${2-$CONFIG} \
--dist