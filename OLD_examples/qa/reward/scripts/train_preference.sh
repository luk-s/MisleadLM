#!/bin/bash
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem="512gb"
#SBATCH --gpus=A100-SXM4-80GB:4
#SBATCH --time="23:59:59"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos="high"
#SBATCH --mail-type=END,FAIL


#TEMP: Maybe remove this: SBATCH --nodelist="cirl.ist.berkeley.edu,rlhf.ist.berkeley.edu,airl.ist.berkeley.edu,sac.ist.berkeley.edu"

export NCCL_P2P_LEVEL=NVL
conda activate mislead
echo "Conda environment: $CONDA_DEFAULT_ENV"

MAX_EPOCH=5

CKPT_NAME=Llama-2-13b-hf

LR=1e-5
# DEEPSPEED=../ds_configs/ds_config_zero2.json
DEEPSPEED=../../ds_configs/ds_config_zero2_custom.json
BC=4
GRAD_ACC=1

let GLOBAL_BATCH_SIZE=8*$BC*$GRAD_ACC
echo "global batch size = "$GLOBAL_BATCH_SIZE

DATA_NAME=arena-55k
TRAIN_DATA=../../data/$DATA_NAME/train_openai.json
VAL_DATA=../../data/$DATA_NAME/test_openai.json

EXP_NAME=lr${LR}_bc${GLOBAL_BATCH_SIZE}_maxepoch${MAX_EPOCH}_full_fixsep


SAVE_DIR=../outputs

LOGGING_DIR=../results/$CKPT_NAME/$EXP_NAME

deepspeed --num_gpus 4 --master_port 6601 ../train_preference.py \
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
    --ckpt_path meta-llama/Llama-2-13b-hf \
    --tokenizer_path meta-llama/Llama-2-13b-hf \
    --gradient_accumulation $GRAD_ACC \
    --flash_attn \
    --eval_steps 50 \
    --save_steps 100