#!/bin/bash
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem="1000gb"
#SBATCH --gpus=A100-SXM4-80GB:8
#SBATCH --time="59:59:59"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos="default"
#SBATCH --mail-type=END,FAIL


# This is required for Multi-GPU training to work
export NCCL_P2P_LEVEL=NVL

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mislead
echo "Conda environment: $CONDA_DEFAULT_ENV"

MAX_EPOCH=10

CKPT_NAME=Llama-2-13b-hf

LR=1e-5
DEEPSPEED=../../configs/ds_config_zero2.json
BC=8
GRAD_ACC=1

TRAIN_DATA=../../data/quality/train.json
VAL_DATA=../../data/quality/test.json

EXP_NAME=lr${LR}_bc${GLOBAL_BATCH_SIZE}_maxepoch${MAX_EPOCH}

SAVE_DIR=XXX

LOGGING_DIR=../results/$CKPT_NAME/$EXP_NAME

deepspeed ../train.py \
    --run_name $EXP_NAME \
    --deepspeed_config $DEEPSPEED \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --output_dir $SAVE_DIR \
    --logging_dir $LOGGING_DIR \
    --max_len 1024 \
    --lr $LR \
    --batch_size $BC \
    --max_epochs $MAX_EPOCH \
    --ckpt_path $CKPT_NAME \
    --gradient_accumulation $GRAD_ACC \
    --flash_attn \
    --eval_steps 50 \
    --save_steps 50