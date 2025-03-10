# This is required for Multi-GPU training to work
export NCCL_P2P_LEVEL=NVL

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mislead
echo "Conda environment: $CONDA_DEFAULT_ENV"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
CUDA_VISIBLE_DEVICES=7 nohup python ../reward_model_general_server.py --reward_model=human_labels_Llama3.2_1B &> ../logging/reward_model/reward_model_general_server_${TIMESTAMP}.log &