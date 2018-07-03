#!/bin/bash

# written and copyright by
# www.github.com/GustavZ


export MODEL="mask_rcnn_mobilenet_v1_300_coco"
export TF_DIR="${HOME}/tf_models/research/object_detection"
export NUM_GPUS=1
export EVAL_ON_CPU=true
export EVAL_ALL_CKPTS=false

export ROOT_DIR="$(pwd)"
export CKPT_DIR="${ROOT_DIR}/checkpoints/${MODEL}/train"
export EVAL_DIR="${ROOT_DIR}/checkpoints/${MODEL}/eval"
export CFG_FILE=${ROOT_DIR}"/configs/"${MODEL}.config

echo "> Infinite Tensorflow Evaluation Loop"
while true; do
    python ${TF_DIR}/eval.py \
        --logtostderr \
        --pipeline_config_path=${CFG_FILE} \
        --checkpoint_dir=${CKPT_DIR} \
        --eval_dir=${EVAL_DIR} \
        --run_on_CPU_only=${EVAL_ON_CPU} \
        --evaluate_all_checkpoints=${EVAL_ALL_CKPTS}
    echo "> sleep 30secs, time to exit"
    sleep 30
done
