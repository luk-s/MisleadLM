import os
import pathlib
from datetime import datetime
from typing import Callable, Dict, List, Optional

from agent_train import DATA_PATH, QADataset, build_metric_fn, build_reward_fn
from transformers import AutoTokenizer

from trlx.data.configs import TRLConfig
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from trlx.trainer.nn.ppo_models import (
    CausalLMHydraWithValueHead,
    Seq2SeqLMHydraWithValueHead,
)
from trlx.utils import set_seed
from trlx.utils.loading import get_pipeline, get_trainer


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
        return CausalLMHydraWithValueHead.from_pretrained(
            config.model.model_path,
            model_architecture,
            num_layers_unfrozen=config.model.num_layers_unfrozen,
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


if __name__ == "__main__":
    # Load the config
    config_path = pathlib.Path(__file__).parent.joinpath("configs/ppo_config_eval.yml")
    config = TRLConfig.load_yaml(config_path)

    # Append the current timestamp to the checkpoint directory
    config.train.checkpoint_dir = (
        f"{config.train.checkpoint_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    # Build the dataset
    # train_path = f"{DATA_PATH}/train_qa.json"
    # test_path = f"{DATA_PATH}/val_qa.json"
    train_path = f"{DATA_PATH}/train_qa_le8000.json"
    test_path = f"{DATA_PATH}/val_qa_le8000.json"
    qa_dataset = QADataset(train_path, test_path)

    # Build the prompts
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    val_prompts = [
        item.build_prompt_for_agent(tokenizer)
        for item in qa_dataset
        if not item.is_train
    ]

    # Load the trainer from a checkpoint
    trainer = build_trainer_for_eval(
        config=config,
        reward_fn=build_reward_fn(qa_dataset),
        metric_fn=build_metric_fn(qa_dataset),
        eval_prompts=val_prompts,  # Typically, eval_prompts can be the same as prompts for evaluation
    )

    # Perform evaluation
    evaluation_results = trainer.evaluate()
    date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    date_time_integer = int(date_time)
    trainer.accelerator.log(evaluation_results, step=date_time_integer)

    # Print evaluation metrics
    print("Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value}")
