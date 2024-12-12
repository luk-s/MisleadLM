## Language Models Learn to Mislead Humans via RLHF

This repository is based on the codebase of the paper:
> [Language Models Learn to Mislead Humans via RLHF](https://arxiv.org/pdf/2409.12822)

and extends it to investigate reward hacking in RLAIF.


### 1. Setup

#### 1.1 Request access to the gated Llama models
- Visit [https://huggingface.co/meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf)
- Log in your huggingface account
- Request access to the model. This step might take ~2 hours.

#### 1.2 Download all files stored with Git Large File Storage
```bash
git lfs install      # Install git lfs
git lfs fetch --all  # Fetch all large files
git lfs checkout     # Teplace the pointer files
```

#### 1.3 Setup the python environment
```bash
conda create -n mislead python=3.10
conda activate mislead
pip install -e .
```

#### 1.4 Log in to necessary services
WeightsAndBiases:
- Follow steps 1 and 2 of this [quickstart](https://docs.wandb.ai/quickstart/).

Huggingface:
- Follow the `Download files` section of [this tutorial](https://huggingface.co/docs/hub/models-gated#download-files)

### 2. RLHF Training 

#### 2.1 Question Answering
```bash
# Train a reward model
cd examples/qa/reward/scripts
bash train_preference.sh # general reward training

# Train an LLM agent against the learned reward model
TODO
```


#### 2.2 Programming
TODO

### 3. Fine-tuned Checkpoints

- [Code generation](https://huggingface.co/jiaxin-wen/MisleadLM-code)
- [Question answering](https://huggingface.co/jiaxin-wen/MisleadLM-QA)