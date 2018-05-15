# tf_training
This repository uses scripts provided by tensorflow's object detection api to train Mask R-CNN with Mobilenet V1 as Backbone on the COCO Dataset.
<br />
Additionally: As many local machine have limited computing capacity (especially GPU Memory)
I added a script that automatically restarts the training and evaluation process from the last saved checkpoint if training got killed by some out of memory / oom or any other error.
> Note: You may need to install several dependencies (you'll face them while running the scripts)

## Getting Started
- inside `dataset/` run `bash download_and_preprocess_mscoco.sh`
- inside `mask_rcnn_mobilenet_v1_coco.config` modify all absolute paths to fit your directory structure
- inside `train.sh` modify the variables `MODEL`, `TF_DIR` and `NUM_GPUS`
- run `train.sh`
- watch the numbers roll!
