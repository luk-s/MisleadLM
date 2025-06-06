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

# Set some variables
MAX_EPOCH=5

CKPT_NAME=Llama-2-13b-hf

LR=1e-5
DEEPSPEED=../../ds_configs/ds_config_zero2_memory_efficient.json
BC=4
GRAD_ACC=1

let GLOBAL_BATCH_SIZE=8*$BC*$GRAD_ACC
echo "global batch size = "$GLOBAL_BATCH_SIZE

DATA_NAME=chatbot-arena
TRAIN_DATA=../../data/$DATA_NAME/evaluation_data.json
VAL_DATA=../../data/$DATA_NAME/evaluation_data.json

EXP_NAME=lr${LR}_bc${GLOBAL_BATCH_SIZE}_maxepoch${MAX_EPOCH}_full_fixsep


SAVE_DIR=../outputs

LOGGING_DIR=../results/$CKPT_NAME/$EXP_NAME

deepspeed --num_gpus 8 --master_port 6601 ../train_preference.py \
# deepspeed --include localhost:2,3,4,5,6,7 --master_port 6601 ../train_preference.py \
    --mode eval \
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
    --ckpt_path ../outputs/human_labels/checkpoint-700/model.safetensors \
    --tokenizer_path meta-llama/Llama-2-13b-hf \
    --gradient_accumulation $GRAD_ACC \
    --flash_attn \
    --eval_steps 50 \
    --save_steps 100