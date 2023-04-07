#!/usr/bin/env bash

NUM_NODES=2
NUM_TRAINERS=2
NODE_RANK=0
MASTER_ADDR=123.456.123 # example IP
MASTER_PORT=1234        # example port
CONFIG=pacbert/config/config.yaml

torchrun \
--nnodes=${1-$NUM_NODES} \
--nproc_per_node=${2-$NUM_TRAINERS} \
--node_rank=${3-$NODE_RANK} \
--master_addr=${4-$MASTER_ADDR} \
--master_port=${5-$MASTER_PORT} \
pacbert/run.py \
--cfg=${6-$CONFIG} \
--dist
