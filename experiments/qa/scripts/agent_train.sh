accelerate launch --main_process_port 25913 --num_processes 7 --config_file ../configs/default_accelerate_config_train.yaml ../agent_train.py | tee ../logging/agent_train.log 2>&1
