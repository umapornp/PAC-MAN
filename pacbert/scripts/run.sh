#!/usr/bin/env bash

CONFIG=pacbert/config/config.yaml

python pacbert/run.py --cfg=${1-$CONFIG}