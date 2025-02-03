#!/bin/bash
#SBATCH --job-name=score_arena_huggingface
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --qos=high

accelerate launch --num_processes 4 --config_file configs/default_accelerate_config.yaml data/arena-55k/create_scorings_huggingface.py