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
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mislead
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Model and data details
MODEL_NAME="meta-llama/Llama-3.1-8B"
# MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL_NAME="/nas/ucb/lukasfluri/data/llama/Llama-3.2-1B-Instruct-hf"
MODEL_NAME_SHORT="${MODEL_NAME##*/}"
TRAIN_DATA="../data/qa/train_qa.json"
NUM_TRAIN_SAMPLES=531
NUM_EVAL_SAMPLES=200

# Training details
NUM_GPUS=8
LR=1e-6
# LR=0
MAX_SEQ_LENGTH=12288 # Note: We don't want any truncation to occur. This value is larger than any tokenized input.
BATCH_SIZE=1
MAX_EPOCHS=5
GRADIENT_ACCUMULATION_STEPS=4
DEEPSPEED_CONFIG="../configs/ds_config_zero2_agent_finetune.json"
#DEEPSPEED_CONFIG="configs/ds_config_zero2_memory_efficient.json"

# Logging details
TIMESTAMP=$(date +"%y-%m-%d_%H:%M:%S")
LOGGING_DIR="../logging/SFT"
let GLOBAL_BATCH_SIZE=$BATCH_SIZE*$GRADIENT_ACCUMULATION_STEPS*$NUM_GPUS
NOW=$(date +"%y-%m-%d_%H:%M:%S")
RUN_NAME="SFT_${MODEL_NAME_SHORT}_lr${LR}_bs${GLOBAL_BATCH_SIZE}_maxepoch${MAX_EPOCHS}_numgpus${NUM_GPUS}_${NOW}"
OUTPUT_DIR="../model_checkpoints/SFT/${RUN_NAME}"
SAVE_STEPS=10
EVAL_STEPS=10

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOGGING_DIR"

echo "Run name: $RUN_NAME"

deepspeed --num_gpus $NUM_GPUS --master_port 6601 ../agent_finetune.py \
    --run_name $RUN_NAME \
    --output_dir $OUTPUT_DIR \
    --model_name $MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --train_data $TRAIN_DATA \
    --num_train_samples $NUM_TRAIN_SAMPLES \
    --num_eval_samples $NUM_EVAL_SAMPLES \
    --lr $LR \
    --max_seq_length $MAX_SEQ_LENGTH \
    --batch_size $BATCH_SIZE \
    --max_epochs $MAX_EPOCHS \
    --gradient_accumulation $GRADIENT_ACCUMULATION_STEPS \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --deepspeed_config $DEEPSPEED_CONFIG 2>&1 | tee "${LOGGING_DIR}/${RUN_NAME}.txt"