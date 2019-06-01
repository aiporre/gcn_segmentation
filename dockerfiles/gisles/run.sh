#!/usr/bin/env bash

mkdir -p /workspace/gfcn/data
ln -s /workspace/container_0/ISLES2018 /workspace/gfcn/data/gisles2018
python training.py -s GISLES2018 -n $NET -b $BATCH -c DCSsigmoid  -N 10 --id $TID -lr $LR -g $EPOCHS --mod TMAX CBF CBV MTT  --postnorm False -D $OUTPUT>> "${OUTPUT}/gfcnb_${TID}.log"

cp -r $OUTPUT /workspace/container_0/$OUTPUT

