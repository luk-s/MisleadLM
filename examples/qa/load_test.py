import argparse
import os

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
)

from trlx.trainer.nn.ppo_models import (
    CausalLMHydraWithValueHead,
)
from trlx.trlx import TRLConfig


# 1. Instantiate the model with the checkpoint and necessary arguments
def load_checkpoint(checkpoint_path, model_path=None, num_layers_unfrozen=-1):
    """
    Load the CausalLMHydraWithValueHead model checkpoint.

    Args:
        checkpoint_path (str): Path to the .bin or state_dict file containing model weights.
        model_path (str or transformers.PretrainedConfig, optional): Path to a config file or a PretrainedConfig.
            If None, the model tries to infer the config from the checkpoint directory.
        num_layers_unfrozen (int, optional): Number of layers to unfreeze in the transformer. Default is -1.

    Returns:
        model: Loaded instance of CausalLMHydraWithValueHead.
    """
    assert model_path is not None, "model_path must be provided"

    # Use config to initialize the model
    model = CausalLMHydraWithValueHead(model_path, num_layers_unfrozen=num_layers_unfrozen)

    # Load the checkpoint weights into the model
    # Load and merge state dicts from all sharded files
    state_dict = {}
    
    # Find the index file that contains shard info
    index_path = f"{checkpoint_path}/pytorch_model.bin.index.json"
    if os.path.exists(index_path):
        import json
        with open(index_path, 'r') as f:
            index_data = json.load(f)
            num_shards = len(index_data['weight_map'].values())

    print(f"num_shards: {num_shards}")

    # Load all shards
    for i in range(1, num_shards + 1):
        shard_path = f"{checkpoint_path}/pytorch_model-{i:05d}-of-0000{num_shards}.bin"
        shard_state_dict = torch.load(shard_path, map_location=torch.device("cpu"))
        state_dict.update(shard_state_dict)
    # Remove any prefix like "module." if the state dict was saved from a distributed training setup
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)  # Use strict=False if there might be extra params
    print(f"Model loaded successfully from {checkpoint_path}")
    

    # Move the model to the appropriate device (e.g., GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)   
    args = parser.parse_args()

    config_path = args.config_path
    config = TRLConfig.load_yaml(config_path)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path, use_fast=False)
    if "Llama-2-" in config.tokenizer.tokenizer_path:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
    elif "Llama-3-" in config.tokenizer.tokenizer_path:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Example usage
    model = load_checkpoint(config.model.model_path, "meta-llama/Llama-2-7b-hf", config.model.num_layers_unfrozen)

    # 3. Verify the model is properly initialized
    print(model)

    # Create a simple test prompt and generate a response
    prompt = "What is the nicest city in France?"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()} # Move inputs to GPU
    outputs = model.generate(**inputs, max_new_tokens=200)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
