#!/bin/bash
set -e  # Exit on error

# Function to clean up on error
cleanup() {
    echo "An error occurred. Cleaning up..."
    tmux kill-session -t replication_run 2>/dev/null || true
}
trap cleanup ERR

echo "Starting setup process..."
echo "Updating repository..."
# Update the repository
cd /nas/ucb/micah/MisleadLM/
git pull

echo "Checking and copying reward model checkpoint..."
# Copy the required reward model checkpoint if it doesn't exist
if [ ! -d "/nas/ucb/micah/MisleadLM/examples/qa/reward/outputs/openai_unbiased_labels" ]; then
    echo "Copying reward model checkpoint..."
    cp -r /nas/ucb/lukasfluri/code/MisleadLM/examples/qa/reward/outputs/openai_unbiased_labels /nas/ucb/micah/MisleadLM/examples/qa/reward/outputs/openai_unbiased_labels
fi

echo "Checking and copying SFT model checkpoint..."
# Copy the required SFT model checkpoint if it doesn't exist
if [ ! -d "/nas/ucb/micah/MisleadLM/examples/qa/outputs/SFT" ]; then
    echo "Copying SFT model checkpoint..."
    cp -r /nas/ucb/lukasfluri/code/MisleadLM/examples/qa/outputs/SFT /nas/ucb/micah/MisleadLM/examples/qa/outputs/SFT
fi

echo "Checking and copying training data..."
# Copy the required training data if it doesn't exist
if [ ! -d "/nas/ucb/micah/MisleadLM/examples/qa/data/qa" ]; then
    echo "Copying training data..."
    cp -r /nas/ucb/lukasfluri/code/MisleadLM/examples/qa/data/qa /nas/ucb/micah/MisleadLM/examples/qa/data/qa
fi

echo "Checking WANDB API key..."
# Check that the wandb API key is set as environment variable
if [ -z "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY is not set. Please set the WANDB_API_KEY environment variable."
    exit 1
fi

echo "Setting up training environment..."
# Go to the correct directory
cd /nas/ucb/micah/MisleadLM/examples/qa

# Properly source Conda for tmux
CONDA_INIT="source $(conda info --base)/etc/profile.d/conda.sh"
$CONDA_INIT
conda activate mislead

# Define the commands
REWARD_MODEL_CMD="(CUDA_VISIBLE_DEVICES=7 nohup python reward_api.py --reward_model=openai_unbiased_labels &> reward_model.log &)"
PPO_CMD="bash train.sh"

# Add NCCL environment variable
NCCL_ENV="export NCCL_P2P_LEVEL=NVL"

# Combine all commands into the srun command
SRUN_CMD="$NCCL_ENV; srun --pty --cpus-per-task=16 --mem=\"1000gb\" --gpus=A100-PCI-80GB --time=\"59:59:59\" --nodes=1 --ntasks-per-node=1 --qos=\"high\" bash -c \"$REWARD_MODEL_CMD && sleep 5 && $PPO_CMD; exec bash\""

echo "Creating and starting tmux session..."
if ! tmux has-session -t replication_run 2>/dev/null; then
    echo "Creating new tmux session..."
    tmux new-session -d -s replication_run
fi

tmux send-keys -t replication_run "$SRUN_CMD" C-m

echo "Setup complete! Started tmux session 'replication_run'. Attach to it using: tmux attach -t replication_run"
