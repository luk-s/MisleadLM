# MODEL_CHECKPOINT="jiaxin-wen/MisleadLM-QA"
MODEL_CHECKPOINT="outputs/ppo_model_openai_unbiased_simple_labels/23680"
CONFIG_FILE="configs/default_accelerate_config_final.yaml"
CONFIG_PATH_TRLX="configs/ppo_config_custom.yml"

accelerate launch --main_process_port 25913 --num_processes 7 --config_file $CONFIG_FILE eval.py --model_path $MODEL_CHECKPOINT --config_path $CONFIG_PATH_TRLX