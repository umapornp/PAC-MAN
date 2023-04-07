#!/usr/bin/env bash

NUM_TRAINERS=2
CONFIG=pacbert/config/config.yaml

torchrun \
--standalone \
--nnodes=1 \
--nproc_per_node=${1-$NUM_TRAINERS} \
pacbert/run.py \
--cfg=${2-$CONFIG} \
--dist