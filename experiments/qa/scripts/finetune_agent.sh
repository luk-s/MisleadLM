#!/bin/bash
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem="1000gb"
#SBATCH --gpus=A100-SXM4-80GB:8
#SBATCH --time="59:59:59"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos="high"
#SBATCH --mail-type=END,FAIL

# This is required for Multi-GPU training to work
export NCCL_P2P_LEVEL=NVL

# Activate the conda environment
conda activate mislead
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Model and data details
MODEL_NAME="meta-llama/Llama-2-7b-hf"
TRAIN_DATA="data/qa/train_qa.json"
NUM_TRAIN_SAMPLES=531
NUM_EVAL_SAMPLES=200

# Training details
NUM_GPUS=8
MAX_LEN=4096
LR=1e-5
BATCH_SIZE=4
MAX_EPOCHS=5
GRADIENT_ACCUMULATION_STEPS=1
DEEPSPEED_CONFIG="configs/ds_config_zero2.json"
#DEEPSPEED_CONFIG="configs/ds_config_zero2_memory_efficient.json"

# Logging details
TIMESTAMP=$(date +"%y-%m-%d_%H:%M:%S")
OUTPUT_DIR="outputs/SFT"
LOGGING_DIR="outputs/SFT/logs"
let GLOBAL_BATCH_SIZE=$BATCH_SIZE*$GRADIENT_ACCUMULATION_STEPS*$NUM_GPUS
RUN_NAME="SFT_${TIMESTAMP}_${MODEL_NAME}_lr${LR}_bs${GLOBAL_BATCH_SIZE}_maxepoch${MAX_EPOCHS}_numgpus${NUM_GPUS}"
SAVE_STEPS=10
EVAL_STEPS=10

deepspeed --num_gpus $NUM_GPUS --master_port 6601 finetune.py \
    --run_name $RUN_NAME \
    --output_dir $OUTPUT_DIR \
    --model_name $MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --train_data $TRAIN_DATA \
    --num_train_samples $NUM_TRAIN_SAMPLES \
    --num_eval_samples $NUM_EVAL_SAMPLES \
    --max_length $MAX_LEN \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --max_epochs $MAX_EPOCHS \
    --gradient_accumulation $GRADIENT_ACCUMULATION_STEPS \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --deepspeed_config $DEEPSPEED_CONFIG