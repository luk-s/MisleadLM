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
conda activate mislead
echo "Conda environment: $CONDA_DEFAULT_ENV"

# MODEL PARAMETERS
MODEL=meta-llama/Llama-3.1-8B-Instruct
MODEL_NAME=${MODEL##*/}

# DATA PARAMETERS
DATA_NAME=arena-55k
TRAIN_DATA=../data/$DATA_NAME/train_human.json
VAL_DATA=../data/$DATA_NAME/test_human.json

# Extract the filename from the training data without the prefix and suffix
TRAIN_DATA_NAME=${TRAIN_DATA##*/}
TRAIN_DATA_NAME=${TRAIN_DATA_NAME%.json}
TRAIN_DATA_NAME=${TRAIN_DATA_NAME#train_}

# TRAINING PARAMETERS 
MAX_EPOCH=5
LR=1e-5
DEEPSPEED=../configs/ds_config_zero2.json
BC=2
GRAD_ACC=1

# LOGGING PARAMETERS
let GLOBAL_BATCH_SIZE=8*$BC*$GRAD_ACC
echo "global batch size = "$GLOBAL_BATCH_SIZE
NOW=$(date +"%Y-%m-%d_%H-%M-%S")
EXP_NAME=MODEL_${MODEL_NAME}_DATA_${TRAIN_DATA_NAME}_LR_${LR}_BC_${GLOBAL_BATCH_SIZE}_MAXEPOCH_${MAX_EPOCH}_TIME_${NOW}

SAVE_DIR=../model_checkpoints/reward_models/$EXP_NAME
LOGGING_DIR=../logging/reward_models/$EXP_NAME
EVAL_STEPS=100
SAVE_STEPS=100


deepspeed --num_gpus 8 --master_port 6601 ../reward_model_general_train.py \
    --run_name $EXP_NAME \
    --deepspeed_config $DEEPSPEED \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --output_dir $SAVE_DIR \
    --logging_dir $LOGGING_DIR \
    --lr $LR \
    --batch_size $BC \
    --max_epochs $MAX_EPOCH \
    --ckpt_path $MODEL \
    --tokenizer_path $MODEL \
    --gradient_accumulation $GRAD_ACC \
    --flash_attn \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS