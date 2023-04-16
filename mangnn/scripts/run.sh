#!/usr/bin/env bash

CONFIG=mangnn/config/config.yaml

python mangnn/run.py --cfg=${1-$CONFIG}