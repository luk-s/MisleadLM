## Language Models Learn to Mislead Humans via RLHF

This repository contains data and code for our paper:
> [Language Models Learn to Mislead Humans via RLHF](https://arxiv.org/pdf/2409.12822)


### 1. Installation
```bash
conda create -n mislead python=3.10
pip install -e .
```

### 2. RLHF Training 

#### 2.1 Programming

```bash
cd src/programming
python reward_api.py
bash train.sh
```

#### 2.2 Question Answering

```bash
cd src/qa/reward
bash train_judge.sh # task-specific reward training
bash train_preference.sh # general reward training

cd ..
CUDA_VISIBLE_DEVICES=6 python reward_api.py # general reward
CUDA_VISIBLE_DEVICES=7 python judge_api.py # task-specific reward
bash train.sh
```