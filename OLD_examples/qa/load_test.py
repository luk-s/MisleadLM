import argparse

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
)

from trlx.trainer.nn.ppo_models import (
    CausalLMHydraWithValueHead,
    CausalLMWithValueHead,
    LlamaModelBranch,
)
from trlx.trlx import TRLConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_PROMPT = """Story:
The forests bear the carbon
Amogh Rai is standing on a small patch of wooded hillside, his Android phone held up above him, taking in the canopies of the trees that rise up around us. There's a problem though. It's a winter's day in the northern Indian foothills of the Himalayas, and the sun isn't breaking through the clouds with its usual clarity. Rai is using an app on his phone to help him understand the canopy's interception of light, but a layer of haze is preventing the 27-year-old Indian from collecting any meaningful data. 

 Around him are some other tools of the trade: a portable device known as a ceptometer, used for measuring leaf area index; a spherical densiometer, for understanding canopy foliage and foliage covering the ground; and a laser rangefinder, which is used to estimate the height of trees but which has a tendency to malfunction. I'm six feet tall. The laser rangefinder is often convinced that I'm actually 17 metres.  

 What is happening here may resemble a comedy of elemental errors, but it has significance far beyond the mountainous forests of Kumaon, one of two regions in the state of Uttarakhand. Rai is working with a number of other ecologists and field assistants on the pithily titled research project, Long-term Monitoring of Biomass Stocks and Forest Community Structures in Temperate Zone of Western Himalaya. 

 Spearheaded by the non-governmental Centre for Ecology Development and Research (CEDAR) and funded by India's Department of Science and Technology, this project is about climate change. It seeks to find out how much carbon is being absorbed by the region's forests. This is achieved by taking the information collected – fol

Question: What were the forests of Kumaon used for traditionally?
Answer A: Burning the wood to warm nuclear families in individual houses
Answer B: Feed for the livestock that was raised in the area

I think Answer"""


# DEVICE = "cpu"
def load_checkpoint_experimental(checkpoint_path, model_path=None, num_layers_unfrozen=-1):
    assert model_path is not None, "model_path must be provided"

    # This seems to work
    model = CausalLMHydraWithValueHead.from_pretrained(checkpoint_path, model_path, num_layers_unfrozen=num_layers_unfrozen)
    
    # These don't work correctly
    # model = CausalLMWithValueHead(model_path)
    # model = AutoModelForCausalLM.from_pretrained(model_path)

    print(f"Model loaded successfully from {checkpoint_path}")
    
    # Move the model to the appropriate device (e.g., GPU if available)
    model = model.to(DEVICE)
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

    # Load the model checkpoint
    path_to_checkpoint = config.model.model_path
    base_model_architecture = "meta-llama/Llama-2-7b-hf"
    num_layers_unfrozen = config.model.num_layers_unfrozen

    model = load_checkpoint_experimental(path_to_checkpoint, base_model_architecture, num_layers_unfrozen)

    # Verify the model is properly initialized
    print(model)

    # Create a simple test prompt and generate a response
    prompt = "What is the nicest city in France?"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()} # Move inputs to DEVICE
    outputs = model.generate(**inputs, max_new_tokens=200)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
