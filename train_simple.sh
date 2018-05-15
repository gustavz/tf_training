#!/bin/bash

# written and copyright by
# www.github.com/GustavZ


export MODEL="mask_rcnn_mobilenet_v1_400_coco"
export TF_DIR="/home/gustav/workspace/tensorflow/tf_models/research/object_detection"

export ROOT_DIR="$(pwd)"
export CKPT_DIR="${ROOT_DIR}/checkpoints/${MODEL}/train"
export EVAL_DIR="${ROOT_DIR}/checkpoints/${MODEL}/eval"
export CFG_FILE=${ROOT_DIR}/${MODEL}.config

echo "> Infinite Tensorflow Training Loop"
while true; do

    # start evaluation in separate terminal
    gnome-terminal -x sh -c "sleep 1m;python ${TF_DIR}/eval.py \
        --logtostderr \
        --pipeline_config_path=${CFG_FILE} \
        --checkpoint_dir=${CKPT_DIR} \
        --eval_dir=${EVAL_DIR}"

    # Start actual training
    echo "> start training ${MODEL}"
    python ${TF_DIR}/train.py \
        --logtostderr  \
        --pipeline_config_path=${CFG_FILE} \
        --train_dir=${CKPT_DIR}

    # wait some time and kill remaining processes
    echo "> waiting 1 minute before restart"
    sleep 60
    killall python
    killall /usr/bin/python
    sleep 30
done
