TIMESTAMP=$(date +"%y-%m-%d_%H:%M:%S")
LOGGING_DIR="../logging/agent"
mkdir -p "$LOGGING_DIR"

accelerate launch --main_process_port 25913 --num_processes 7 --config_file ../configs/default_accelerate_config_train.yaml ../agent_train.py 2>&1 | tee "$LOGGING_DIR/agent_train_$TIMESTAMP.log"
