TIMESTAMP=$(date +"%y-%m-%d_%H:%M:%S")
LOGGING_DIR="../logging/agent"
mkdir -p "$LOGGING_DIR"

# The PPO agent might have to wait quite a long time for the reward model (that runs in a separate process)
# to score all responses during evaluation. If we don't increase this timeout, the CUDA watchdog will get
# impatient and kill the process.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

accelerate launch --main_process_port 25913 --num_processes 7 --config_file ../configs/default_accelerate_config_train.yaml ../agent_train.py 2>&1 | tee "$LOGGING_DIR/agent_train_$TIMESTAMP.log"
