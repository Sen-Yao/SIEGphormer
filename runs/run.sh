#!/bin/bash

cmd_time=`TZ=UTC-8 date  "+%Y%m%d-%H%M%S"`

nohup python ./src/run.py \
  --data_name cora \
  --lr 1e-3 \
  --gnn-layers 1 \
  --dim 128 \
  --batch-size 1024 \
  --epochs 200 \
  --eps 1e-7 \
  --gnn-drop 0.1 \
  --dropout 0.1 \
  --pred-drop 0.1 \
  --att-drop 0.1 \
  --num-heads 1 \
  --thresh-1hop 1e-2 \
  --thresh-non1hop 1e-2 \
  --feat-drop 0.1 \
  --l2 0 \
  --eval_steps 1 \
  --decay 0.975 \
  --runs 10 \
  --kill_cnt 30 \
  --mat_prop 1 \
  --alpha 0.7 \
  --drnl 1 \
  --device 0 \
> ./logs/${cmd_time}-debug.log 2>&1 &