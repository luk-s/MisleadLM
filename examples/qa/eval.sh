# Define GPU IDs as a comma-separated list
GPU_IDS="0,1,2,3,4,5,6,7"  # You can modify this to use different GPUs, e.g. "0" or "0,1,2"

# Calculate number of processes based on number of GPUs
NUM_PROCESSES=$(echo $GPU_IDS | tr ',' '\n' | wc -l)

CONFIG_FILE="configs/default_accelerate_config_final.yaml"
CONFIG_PATH_TRLX="configs/ppo_config_eval.yml"

accelerate launch \
    --main_process_port 25913 \
    --num_processes $NUM_PROCESSES \
    --gpu_ids $GPU_IDS \
    --config_file $CONFIG_FILE \
    eval.py --config_path $CONFIG_PATH_TRLX