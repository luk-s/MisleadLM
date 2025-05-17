# This is required for Multi-GPU training to work
export NCCL_P2P_LEVEL=NVL

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mislead
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Define GPU IDs as a comma-separated list
GPU_IDS="0,1,2,3,4,5,6,7"  # You can modify this to use different GPUs, e.g. "0" or "0,1,2"

# Calculate number of processes based on number of GPUs
NUM_PROCESSES=$(echo $GPU_IDS | tr ',' '\n' | wc -l)

CONFIG_FILE_ACCELERATE="configs/default_accelerate_config_final.yaml"

# Build all arguments
USE_LEGACY_FORMAT=true
COMPUTE_REWARD_MODEL_SCORES=false
USE_TRAINER_EVALUATION=true
MODEL_ARCHITECTURE="meta-llama/Llama-2-7b-hf"
TRAIN_PATH="../data/qa/train_qa.json"
VALIDATION_PATH="../data/qa/val_qa.json"
CONFIG_FILE_TRLX="../configs/ppo_config_eval_legacy.yml"

ARGS="--config_path $CONFIG_FILE_TRLX --train_path $TRAIN_PATH --validation_path $VALIDATION_PATH --model_architecture $MODEL_ARCHITECTURE"
if [ "$USE_LEGACY_FORMAT" = true ]; then
    ARGS="$ARGS --legacy_format"
fi
if [ "$COMPUTE_REWARD_MODEL_SCORES" = true ]; then
    ARGS="$ARGS --compute_reward_model_scores"
fi
if [ "$USE_TRAINER_EVALUATION" = true ]; then
    ARGS="$ARGS --use_trainer_evaluation"
fi

accelerate launch \
    --main_process_port 25913 \
    --num_processes $NUM_PROCESSES \
    --gpu_ids $GPU_IDS \
    --config_file $CONFIG_FILE_ACCELERATE \
    ../agent_eval.py $ARGS