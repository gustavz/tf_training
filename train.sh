#!/bin/bash

# written and copyright by
# www.github.com/GustavZ


export MODEL="mask_rcnn_mobilenet_v1_512_coco"
export TF_DIR="${HOME}/workspace/tf_models/research/object_detection"
export NUM_GPUS=2
export EVAL_ON_CPU=true
export FIRST_RUN=false

export ROOT_DIR="$(pwd)"
export CKPT_DIR="${ROOT_DIR}/checkpoints/${MODEL}/train"
export EVAL_DIR="${ROOT_DIR}/checkpoints/${MODEL}/eval"
export CFG_FILE=${ROOT_DIR}"/configs/"${MODEL}.config

echo "> Infinite Tensorflow Training Loop"
while true; do
    ## check if first run
    if [ ${FIRST_RUN} = false ] ; then
        echo "> update checkpoint"
        # find old checkpoint
        old=`sed -n '/fine_tune_checkpoint/p' ${CFG_FILE}` # find checkpoint line
        old=${old#*"model."} #strip prefix
        old=${old%\"} #strip suffix
        echo "> old: ${old}"
        # find latest checkpoint
        unset -v latest
        for file in ${CKPT_DIR}/*".meta"; do
          [[ $file -nt $latest ]] && latest=$file
        done
        latest=${latest#*"model."} #strip prefix
        latest=${latest%".meta"} #strip suffix
        echo "> latest: ${latest}"
        # update config
        sed -i s/${old}/${latest}/g ${CFG_FILE}
    else
        export FIRST_RUN=false
    fi

    # Tensorboard % Evaluation
    echo "> start tensorboard and eval.py in separate terminals with 1m delay"
    #gnome-terminal -x sh -c "sleep 1m;tensorboard --logdir=${ROOT_DIR}"
    gnome-terminal -x sh -c "sleep 1m;python ${TF_DIR}/eval.py \
        --logtostderr \
        --pipeline_config_path=${CFG_FILE} \
        --checkpoint_dir=${CKPT_DIR} \
        --eval_dir=${EVAL_DIR} \
        --run_on_CPU_only=${EVAL_ON_CPU}"

    # Start actual training
    echo "> start training ${MODEL}"
    python ${TF_DIR}/train.py \
        --logtostderr  \
        --pipeline_config_path=${CFG_FILE} \
        --train_dir=${CKPT_DIR} \
        --num_clones=${NUM_GPUS} --ps_tasks=1

    # wait some time and kill remaining processes
    echo "> waiting 1 minute before restart"

    sleep 30
    killall python
    killall /usr/bin/python
    sleep 30
done
