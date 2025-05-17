import argparse
import os
import pathlib
from datetime import datetime
from typing import Callable, Dict, List, Optional

import torch
from agent_train import build_metric_fn, build_reward_fn
from huggingface_hub import HfApi
from qa_dataset import QADataset
from tqdm import tqdm
from transformers import AutoTokenizer

from trlx.data.configs import TRLConfig
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from trlx.trainer.nn.ppo_models import (
    CausalLMHydraWithValueHead,
    Seq2SeqLMHydraWithValueHead,
)
from trlx.utils import set_seed
from trlx.utils.loading import get_pipeline, get_trainer

CURRENT_DIR = pathlib.Path(__file__).parent


def build_trainer_for_eval(
    config: TRLConfig,
    reward_fn: Optional[
        Callable[[List[str], List[str], List[str]], List[float]]
    ] = None,
    metric_fn: Optional[
        Callable[[List[str], List[str], List[str]], Dict[str, List[float]]]
    ] = None,
    eval_prompts: Optional[List[str]] = None,
    stop_sequences: Optional[List[str]] = [],
    model_architecture: str = "meta-llama/Llama-2-7b-hf",
) -> AccelerateRLTrainer:
    """
    Creates a trainer object for evaluation purposes

    Args:
        reward_fn (Optional[Callable]): Function to compute evaluation metrics
        metric_fn (Optional[Callable]): Function to compute evaluation metrics
        eval_prompts (List[str]): Prompts to use for evaluation
        config (Optional[TRLConfig]): TRLX configuration object
        stop_sequences (Optional[List[str]]): String sequences to trim generations

    Returns:
        AccelerateRLTrainer: The trainer object configured for evaluation
    """

    set_seed(config.train.seed)

    # Super hacky way to get the trainer to use the correct model loader during it's __init__ method
    def get_arch(self, config: TRLConfig):
        if config.model.model_arch_type == "seq2seq":
            return Seq2SeqLMHydraWithValueHead(
                config.model.model_path, config.model.num_layers_unfrozen
            )

        # Check whether the provided model path is a hf repo
        hf_api = HfApi()
        if hf_api.repo_exists(repo_id=config.model.model_path, repo_type="model"):
            # Create a name for the local directory
            local_dir = (
                CURRENT_DIR
                / "model_checkpoints/PPO"
                / config.model.model_path.replace("/", "_")
            ).absolute()
        else:
            local_dir = None

        return CausalLMHydraWithValueHead.from_pretrained(
            config.model.model_path,
            model_architecture,
            num_layers_unfrozen=config.model.num_layers_unfrozen,
            hf_local_target_dir=local_dir,
        )

    trainer_class = get_trainer(config.train.trainer)
    trainer_class.get_arch = get_arch
    trainer = trainer_class(
        config=config,
        reward_fn=reward_fn,
        metric_fn=metric_fn,
        stop_sequences=stop_sequences,
        **config.train.trainer_kwargs,
    )

    assert trainer.metric_fn, "metric_fn is required"

    batch_size = config.train.batch_size * int(os.environ.get("WORLD_SIZE", 1))
    max_prompt_length = (
        config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]
    )

    if eval_prompts is None:
        eval_prompts = [trainer.tokenizer.bos_token] * batch_size

    if config.train.pipeline == "GLMPromptPipeline":
        eval_pipeline = get_pipeline(config.train.pipeline)(
            eval_prompts,
            max_prompt_length,
            config.method.gen_kwargs["max_new_tokens"],
            trainer.tokenizer,
        )
    else:
        eval_pipeline = get_pipeline(config.train.pipeline)(
            eval_prompts, max_prompt_length, trainer.tokenizer
        )
    trainer.add_eval_pipeline(eval_pipeline)

    # trainer.learn()

    # Taken from the trainer.learn() function
    trainer.generate_sweep_kwarg = None
    for k, v in config.method.gen_kwargs.items():
        if isinstance(v, list):
            if trainer.generate_sweep_kwarg is not None:
                print(
                    f"Only a single sweep is allowed, {k} is going to be set to {v[0]}"
                )
                trainer.generate_kwargs[k] = v[0]
            else:
                trainer.generate_sweep_kwarg = (k, v)

    trainer.iter_count = 0
    trainer.nth_evaluation = 0

    # Taken from the trainer.prepare_learning() function
    eval_dataloader = trainer.eval_pipeline.create_loader(config.train.eval_batch_size)
    trainer.eval_dataloader = trainer.accelerator.prepare_data_loader(eval_dataloader)
    trainer.n_inner_epochs = trainer.config.method.ppo_epochs
    trainer.total_steps = (
        trainer.config.train.epochs
        * trainer.n_inner_epochs
        * config.method.num_rollouts
    )
    trainer.total_steps = min(trainer.total_steps, trainer.config.train.total_steps)

    return trainer


def evaluate_agent_with_trainer_pipeline(
    config: TRLConfig,
    qa_dataset: QADataset,
    compute_reward_model_scores: bool = False,
    model_architecture: str = "meta-llama/Llama-2-7b-hf",
) -> Dict[str, List[float]]:
    """
    Evaluates the agent by using the trainer pipeline of the 'trlx' library that has been customized by the authors of the original codebase.

    Note: If 'compute_reward_model_scores' is set to True, it is assumed that the reward model has been started as a separate process and is listening on the correct port.

    Args:
        config (TRLConfig): The configuration object
        qa_dataset (QADataset): The dataset object
        compute_reward_model_scores (bool): Whether to compute reward model scores
    Returns:
        Dict[str, List[float]]: A dictionary containing the evaluation results
    """
    # Build the prompts
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    evaluation_prompts = [
        item.build_prompt_for_agent(tokenizer)
        for item in qa_dataset
        if not item.is_train
    ]

    # Load the trainer from a checkpoint
    trainer = build_trainer_for_eval(
        config=config,
        reward_fn=build_reward_fn(
            dataset=qa_dataset,
            tokenizer=tokenizer,
            skip_start_and_end_tokens=True,
            compute_reward_model_scores=compute_reward_model_scores,
        ),
        metric_fn=build_metric_fn(
            dataset=qa_dataset,
            tokenizer=tokenizer,
            skip_start_and_end_tokens=True,
            compute_reward_model_scores=compute_reward_model_scores,
        ),
        eval_prompts=evaluation_prompts,  # Typically, eval_prompts can be the same as prompts for evaluation
        model_architecture=model_architecture,
    )

    evaluation_results = trainer.evaluate()
    date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    date_time_integer = int(date_time)
    trainer.accelerator.log(evaluation_results, step=date_time_integer)

    return evaluation_results


def evaluate_agent_manually(
    config: TRLConfig,
    qa_dataset: QADataset,
    compute_reward_model_scores: bool = False,
    model_architecture: str = "meta-llama/Llama-2-7b-hf",
) -> Dict[str, List[float]]:
    # Check whether the provided model path is a hf repo
    hf_api = HfApi()
    if hf_api.repo_exists(repo_id=config.model.model_path, repo_type="model"):
        # Create a name for the local directory
        local_dir = (
            CURRENT_DIR
            / "model_checkpoints/PPO"
            / config.model.model_path.replace("/", "_")
        ).absolute()
    else:
        local_dir = None

    # Load the model
    model = CausalLMHydraWithValueHead.from_pretrained(
        config.model.model_path,
        model_architecture,
        num_layers_unfrozen=config.model.num_layers_unfrozen,
        hf_local_target_dir=local_dir,
    )

    # Build the prompts
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    evaluation_prompts = [
        item.build_prompt_for_agent(tokenizer)
        for item in qa_dataset
        if not item.is_train
    ]
    evaluation_prompts = evaluation_prompts[:10]

    max_prompt_length = (
        config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]
    )

    # Generate the responses
    responses = []
    with torch.no_grad():
        for prompt in tqdm(evaluation_prompts, desc="Generating responses"):
            model_inputs = tokenizer(
                prompt,
                truncation=True,
                padding=False,
                max_length=max_prompt_length,
                add_special_tokens=True,
                return_tensors="pt",
            )

            responses.append(
                model.generate(
                    input_ids=model_inputs["input_ids"].to(model.device),
                    attention_mask=model_inputs["attention_mask"].to(model.device),
                    **config.method.gen_kwargs,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            )

    # Compute the metrics
    metric_function = build_metric_fn(
        dataset=qa_dataset,
        tokenizer=tokenizer,
        skip_start_and_end_tokens=True,
        compute_reward_model_scores=compute_reward_model_scores,
    )
    metrics = metric_function(responses)

    return metrics


if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Evaluate RL agent on QA tasks")

    # fmt: off
    parser.add_argument("--legacy_format", action="store_true", default=False, help="Use legacy data format and model configuration")
    parser.add_argument("--compute_reward_model_scores", action="store_true", default=False, help="Compute reward model scores (requires reward model server)")
    parser.add_argument("--model_architecture", type=str, help="The name of the underlying architecture of the model under evaluation. Should correspond to a model on Hugging Face.")
    parser.add_argument("--use_trainer_evaluation", action="store_true", default=False, help="Use trainer evaluation pipeline instead of manual evaluation")
    parser.add_argument("--config_path", type=str, help="Path to custom config file.")
    parser.add_argument("--train_path", type=str, help="Path to training data.")
    parser.add_argument("--validation_path", type=str, help="Path to validation data.")
    # fmt: on

    args = parser.parse_args()

    # Set configuration based on command-line arguments
    config_path = args.config_path

    # Load the config
    config = TRLConfig.load_yaml(config_path)

    # Append the current timestamp to the checkpoint directory
    config.train.checkpoint_dir = (
        f"{config.train.checkpoint_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    # Build the dataset
    qa_dataset = QADataset(
        args.train_path, args.validation_path, use_legacy_format=args.legacy_format
    )

    # Perform evaluation
    if args.use_trainer_evaluation:
        evaluation_results = evaluate_agent_with_trainer_pipeline(
            config,
            qa_dataset,
            args.compute_reward_model_scores,
            args.model_architecture,
        )
    else:
        evaluation_results = evaluate_agent_manually(
            config,
            qa_dataset,
            args.compute_reward_model_scores,
            args.model_architecture,
        )

    # Print evaluation metrics
    print("Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value}")
