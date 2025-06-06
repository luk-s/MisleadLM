import json
import os
import uuid
from time import time
from typing import Callable, List, Optional

import numpy as np
import ray
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import trlx.utils.logging as logging
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.configs import TRLConfig
from trlx.data.ppo_types import GLMPPORLBatch, GLMPPORLElement, PPORLBatch, PPORLElement
from trlx.pipeline.offline_pipeline import PromptPipeline
from trlx.pipeline.ppo_pipeline import GLMPPORolloutStorage, PPORolloutStorage
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from trlx.trainer.nn.ppo_models import (
    AdaptiveKLController,
    CausalLMHydraWithValueHead,
    FixedKLController,
    Seq2SeqLMHydraWithValueHead,
)
from trlx.utils import Clock
from trlx.utils.modeling import RunningMoments, logprobs_of_labels

logger = logging.get_logger(__name__)


@register_trainer
class AcceleratePPOTrainer(AccelerateRLTrainer):
    """PPO Accelerate Trainer"""

    reward_fn: Callable[[List[str], List[str], List[str]], List[float]]
    tokenizer: AutoTokenizer

    def __init__(self, config: TRLConfig, **kwargs):
        """PPO Accelerate Trainer initialization

        Args:
            config: Config
        """
        super().__init__(config, **kwargs)

        # Setup rollout logging
        if config.train.rollout_logging_dir is not None:
            self.log_rollouts = True
            self.setup_rollout_logging(config)
        else:
            self.log_rollouts = False

        # Setup the rollout store
        # Rollouts contain the prompt & response, log probs, values and rewards - from each rollout
        if self.tokenizer.__class__.__name__ == "GLMGPT2Tokenizer":
            self.store = GLMPPORolloutStorage(self.tokenizer.pad_token_id)
        else:
            self.store = PPORolloutStorage(self.tokenizer.pad_token_id)

        # Create the rollout store dataloader (for batching up rollouts)
        # TODO (jon-tow): This is only used to satisfy to `accelerator.prepare` call constraint below - remove in future
        rollout_loader: DataLoader = self.store.create_loader(
            self.config.train.batch_size, shuffle=True
        )

        # Prepare multi-GPU acceleration
        self.model, self.opt, self.scheduler, rollout_loader = self.accelerator.prepare(
            self.model, self.opt, self.scheduler, rollout_loader
        )

        self.store.clear_history()  # Clear the rollout store

        # Setup a reference model when hydra heads are not used
        if not hasattr(self.model, "frozen_head"):
            self.ref_model = self.get_arch(self.config)
            self.ref_model.to(self.accelerator.device)

        # Setup the KL controller
        # This helps prevent large divergences in the controller (policy)
        print("target = ", config.method.target)
        # assert config.method.target == 'None'
        if config.method.target != "None":
            self.kl_ctl = AdaptiveKLController(
                config.method.init_kl_coef, config.method.target, config.method.horizon
            )
        else:
            self.kl_ctl = FixedKLController(config.method.init_kl_coef)

        # Create the parameters for the Hugging Face language model's generator
        # method (that generates new tokens from a prompt).
        # https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/text_generation#transformers.GenerationMixin.generate
        if config.model.model_arch_type == "seq2seq":
            self.generate_kwargs = dict(
                config.method.gen_kwargs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if config.method.gen_experience_kwargs is not None:
                self.generate_experience_kwargs = dict(
                    config.method.gen_experience_kwargs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            else:
                self.generate_experience_kwargs = None
        else:
            self.generate_kwargs = dict(
                config.method.gen_kwargs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if config.method.gen_experience_kwargs is not None:
                self.generate_experience_kwargs = dict(
                    config.method.gen_experience_kwargs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            else:
                self.generate_experience_kwargs = None

        # Setup stats tracker
        self.running_moments = RunningMoments()
        self.ref_mean = self.config.method.ref_mean
        self.ref_std = self.config.method.ref_std

    def get_arch(self, config: TRLConfig):
        """Get the model"""
        if config.model.model_arch_type == "seq2seq":
            return Seq2SeqLMHydraWithValueHead(
                config.model.model_path, config.model.num_layers_unfrozen
            )
        return CausalLMHydraWithValueHead(
            config.model.model_path, config.model.num_layers_unfrozen
        )

    def loss(self, batch):
        """Forward pass & loss

        Args:
            batch: Previous batch of episodes
        """
        # Move `batch` data to `accelerator` device
        if self.tokenizer.__class__.__name__ == "GLMGPT2Tokenizer":
            query_strs = batch.query
            response_strs = batch.response
            response_tensors = batch.response_tensors.to(self.accelerator.device)
        else:
            query_tensors = batch.query_tensors.to(self.accelerator.device)
            response_tensors = batch.response_tensors.to(self.accelerator.device)
        old_logprobs = batch.logprobs.to(self.accelerator.device)
        old_values = batch.values.to(self.accelerator.device)
        old_rewards = batch.rewards.to(self.accelerator.device)
        response_length = old_rewards.shape[1]

        advantages, returns = self.config.method.get_advantages_and_returns(
            old_values, old_rewards, response_length
        )

        if self.config.model.model_arch_type == "seq2seq":
            input_ids = query_tensors
            decoder_input_ids = response_tensors
            attention_mask = (
                input_ids.ne(self.tokenizer.pad_token_id)
                .long()
                .to(self.accelerator.device)
            )

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )

            logits = outputs.logits
            values_pred = outputs.value
            logprobs = logprobs_of_labels(logits[:, :-1, :], decoder_input_ids[:, 1:])
            mask = (
                decoder_input_ids.ne(self.tokenizer.pad_token_id)
                .long()
                .to(self.accelerator.device)
            )
            start = 1
            end = start + response_length
            logprobs, values_pred, mask = (
                logprobs[:, start:end],
                values_pred[:, start - 1 : end - 1],
                mask[:, start:end],
            )
        else:
            if self.tokenizer.__class__.__name__ == "GLMGPT2Tokenizer":
                max_prompt_length = (
                    self.config.train.seq_length
                    - self.config.method.gen_kwargs["max_new_tokens"]
                )
                inputs = self.tokenizer(
                    query_strs,
                    truncation=True,
                    padding=True,
                    max_length=max_prompt_length,
                    return_tensors="pt",
                )
                query_tensors = inputs.input_ids
                inputs = self.tokenizer.build_inputs_for_generation(
                    inputs,
                    targets=response_strs,
                    max_gen_length=self.config.method.gen_kwargs["max_new_tokens"] + 1,
                )
                tokens = inputs.input_ids.to(self.accelerator.device)
                attention_mask = inputs.attention_mask.to(self.accelerator.device)
                position_ids = inputs.position_ids.to(self.accelerator.device)
                outputs = self.model(
                    tokens,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    return_dict=True,
                )
                logits = outputs.logits
                values_pred = outputs.value
                values_pred = values_pred[:, :-1]
                logprobs = logprobs_of_labels(logits[:, :-1, :], tokens[:, 1:])

                mask = (
                    response_tensors.ne(self.tokenizer.pad_token_id)
                    .long()
                    .to(self.accelerator.device)
                )
                start = query_tensors.shape[1]
                end = start + response_length

                logprobs, values_pred, mask = (
                    logprobs[:, start:end],
                    values_pred[:, start:end],
                    mask[:, :response_length],
                )
            else:
                tokens = torch.cat((query_tensors, response_tensors), dim=1)
                attention_mask = (
                    tokens.not_equal(self.tokenizer.pad_token_id)
                    .long()
                    .to(tokens.device)
                )
                outputs = self.model(tokens, attention_mask, return_dict=True)
                logits = outputs.logits
                values_pred = outputs.value
                values_pred = values_pred[:, :-1]
                logprobs = logprobs_of_labels(logits[:, :-1, :], tokens[:, 1:])

                start = query_tensors.shape[1] - 1
                end = start + response_length
                logprobs, values_pred, mask = (
                    logprobs[:, start:end],
                    values_pred[:, start:end],
                    attention_mask[:, start:end],
                )

        loss, stats = self.config.method.loss(
            logprobs=logprobs,
            values=values_pred,
            old_logprobs=old_logprobs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            mask=mask,
        )
        return loss, stats

    def setup_rollout_logging(self, config):
        # Make rollout logging dir for this run and store config
        exists = os.path.exists(config.train.rollout_logging_dir)
        isdir = os.path.isdir(config.train.rollout_logging_dir)
        assert exists and isdir

        self.run_id = f"run-{uuid.uuid4()}"
        self.rollout_logging_dir = os.path.join(
            config.train.rollout_logging_dir, self.run_id
        )
        os.mkdir(self.rollout_logging_dir)

        with open(os.path.join(self.rollout_logging_dir, "config.json"), "w") as f:
            f.write(json.dumps(config.to_dict(), indent=2))

    def post_epoch_callback(self):
        """Post epoch callback

        Clears the store and creates `num_rollouts` new episodes.
        """
        if self.log_rollouts:
            self.store.export_history(location=self.rollout_logging_dir)
        self.store.clear_history()
        # Collect more rollouts for training
        self.make_experience(self.config.method.num_rollouts, self.iter_count)

    def post_backward_callback(self):
        self.kl_ctl.update(self.mean_kl, n_steps=self.config.train.batch_size)

    def create_train_dataloader(self):
        return self.store.create_loader(self.config.train.batch_size, shuffle=True)

    def prepare_learning(self):
        eval_dataloader = self.eval_pipeline.create_loader(
            self.config.train.eval_batch_size
        )
        self.eval_dataloader = self.accelerator.prepare_data_loader(eval_dataloader)
        # self.train_dataloader = self.store.create_loader(self.config.train.batch_size, shuffle=True)
        self.train_dataloader = self.create_train_dataloader()
        # self.n_updates_per_batch = self.config.method.ppo_epochs
        # self.total_steps = self.config.train.epochs * self.n_updates_per_batch * len(self.train_dataloader)
        self.n_inner_epochs = self.config.method.ppo_epochs
        self.total_steps = (
            self.config.train.epochs * self.n_inner_epochs * len(self.train_dataloader)
        )
        self.total_steps = min(self.total_steps, self.config.train.total_steps)

    def add_prompt_pipeline(self, pipeline):
        """Add a prompt pipeline dataloader to a trainer instance for the `make_experience` stage"""
        prompt_dataloader = pipeline.create_loader(
            self.config.method.chunk_size, shuffle=True
        )
        self.prompt_dataloader = self.accelerator.prepare_data_loader(prompt_dataloader)
        self.prompt_iterator = iter(self.prompt_dataloader)

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):  # noqa:
        """Make experiences

        Takes `chunk_size` number of prompts from `prompt_iterator`, samples
        from the model and then computes the KL against a reference model. Finally it
        then appends PPOElements to trainer's `store`.

        Args:
            num_rollouts: Number of rollouts to generate
            iter_count: Total number of updates run (i.e. number of updates run for all batches & epochs)
        """
        logger.info("Collecting rollouts")
        tbar = logging.tqdm(
            total=num_rollouts,
            disable=os.environ.get("RANK", 0) != "0",
            desc=f"[rollout 0 / {num_rollouts}]",
            # Lower progress bar by 1 if we're in WARNING mode or above to avoid hiding high priority progress
            # bars (e.g. loss progress in trainers)
            position=logging.get_verbosity() >= logging.WARNING,
            # Leave progress bar if we're in INFO mode or lower to avoid spamming in suppressed verbosity levels
            leave=logging.get_verbosity() < logging.WARNING,
        )

        clock = Clock()
        ppo_rl_elements = []
        accumulate_stats = []

        while len(ppo_rl_elements) < num_rollouts:
            stats = {}

            # Get next batch in prompt dataset and refresh if exhausted
            # TOOD (jon-tow): Make `prompt_dataloader` a cyclic/infinite DataLoader to not require manually
            # "refreshing" the contents of the `prompt_iterator`
            try:
                batch: PromptBatch = next(self.prompt_iterator)
            except StopIteration:
                self.prompt_iterator = iter(self.prompt_dataloader)
                batch = next(self.prompt_iterator)

            exp_generate_time = time()
            # Generate samples from the language model (similar to using HuggingFace `generate` method)
            samples = self.generate(**batch)
            stats["time/exp_generate"] = time() - exp_generate_time

            prompt_tensors = batch.input_ids
            device = samples.device

            prompt_sizes = torch.tensor(
                [prompt_tensors.shape[1]] * len(prompt_tensors), device=device
            )
            padded_samples = self.accelerator.pad_across_processes(
                samples,
                dim=1,
                pad_index=self.tokenizer.eos_token_id
                if "generation_attention_mask" not in batch
                else self.tokenizer.eop_token_id,
                pad_first=False,
            )
            padded_prompts = self.accelerator.pad_across_processes(
                prompt_tensors,
                dim=1,
                pad_index=self.tokenizer.eos_token_id,
                pad_first=False,
            )
            gathered_samples = self.accelerator.gather(padded_samples)
            gathered_prompts = self.accelerator.gather(padded_prompts)
            gathered_prompt_sizes = self.accelerator.gather(prompt_sizes)

            if self.accelerator.is_main_process:
                all_str_samples, all_str_prompts, all_str_outputs = self.decode(
                    gathered_prompts, gathered_samples, gathered_prompt_sizes
                )
                exp_score_time = time()
                all_scores = (
                    self.reward_fn(
                        samples=all_str_samples,
                        prompts=all_str_prompts,
                        outputs=all_str_outputs,
                    )
                    .clone()
                    .detach()
                    .to(device)
                )

                stats["time/exp_score"] = time() - exp_score_time
                all_scores = list(
                    all_scores.reshape(self.accelerator.num_processes, -1).unbind()
                )
            else:
                all_scores = None

            # use torch 1.13.1
            if torch.distributed.is_initialized():
                scores = torch.empty(len(samples), device=device)
                torch.distributed.scatter(scores, all_scores)
            else:
                scores = torch.tensor(all_scores[0])

            str_samples, str_prompts, str_outputs = self.decode(prompt_tensors, samples)

            # Pad the sample outputs
            outputs = self.tokenizer(
                [i + self.tokenizer.eos_token for i in str_outputs],
                add_special_tokens=False,
            ).input_ids
            outputs = list(map(torch.LongTensor, outputs))
            maxsize = max(map(len, outputs))
            outputs = [
                F.pad(
                    output,
                    (0, maxsize - len(output)),
                    value=self.tokenizer.pad_token_id,
                )
                for output in outputs
            ]
            sample_outputs = torch.vstack(outputs).to(device)

            # store statistics of the initial rollout as reference

            clip_reward = self.config.method.cliprange_reward
            if clip_reward:
                scores = torch.clip(scores, -clip_reward, clip_reward)

            if self.ref_mean is None:
                self.ref_mean, self.ref_std = scores.mean(), scores.std()
            all_scores_mean, all_scores_std = self.running_moments.update(scores)
            stats["exp_scores/mean"] = all_scores_mean
            stats["exp_scores/std"] = all_scores_std
            stats["exp_scores/running_mean"] = self.running_moments.mean
            stats["exp_scores/running_std"] = self.running_moments.std

            if self.config.method.scale_reward == "running":
                scores /= self.running_moments.std
            elif self.config.method.scale_reward == "ref":
                scores /= self.ref_std

            # Precompute logprobs, values
            if self.config.model.model_arch_type == "seq2seq":
                attention_mask = batch.attention_mask.to(device)
                prompt_tensors = batch.input_ids.to(device)
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=prompt_tensors,
                        attention_mask=attention_mask,
                        decoder_input_ids=sample_outputs,
                    )
                    logits = outputs.logits
                    values = outputs.value
                    if hasattr(self.model, "frozen_head"):
                        ref_logits = self.model.forward_hydra(
                            input_ids=prompt_tensors,
                            attention_mask=attention_mask,
                            decoder_input_ids=sample_outputs,
                        )
                    else:
                        ref_logits = self.ref_model(
                            input_ids=prompt_tensors,
                            attention_mask=attention_mask,
                            decoder_input_ids=sample_outputs,
                        ).logits
            else:
                all_tokens = torch.cat(
                    (prompt_tensors.to(device), sample_outputs), dim=1
                )
                attention_mask = (
                    all_tokens.not_equal(self.tokenizer.pad_token_id).long().to(device)
                )
                position_ids = None
                with torch.no_grad():
                    logits, *_, values = self.model(
                        all_tokens,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )

                    # Optimize GPU memory usage
                    logits = logits.cpu()
                    values = values.cpu()
                    torch.cuda.empty_cache()

                    # TODO(dahoas): When hydra model works need to also support generation on hydra head
                    if hasattr(self.model, "frozen_head"):
                        ref_logits = self.model.forward_hydra(
                            all_tokens,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            return_dict=False,
                        )
                    else:
                        ref_logits, _, *_ = self.ref_model(
                            all_tokens,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            return_dict=False,
                        )
                        ref_logits = ref_logits.to(device)

            all_tokens = all_tokens.cpu()
            attention_mask = attention_mask.cpu()
            ref_logits = ref_logits.cpu()
            logprobs = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:]).cpu()
            ref_logprobs = logprobs_of_labels(
                ref_logits[:, :-1, :], all_tokens[:, 1:]
            ).cpu()

            n_samples: int = samples.shape[0]

            log_ratio = (logprobs - ref_logprobs) * attention_mask[:, :-1]
            kl = log_ratio.exp() - 1 - log_ratio
            mean_kl = kl.sum(1).mean()  # sequence-wise KL

            # Estimate the KL divergence between the model and reference model
            if self.config.model.model_arch_type == "seq2seq":
                # Skip the beginning of sequence token
                start = 1

                # Get the number of non-padding tokens for each sample
                # This assumes all padding is on the right side
                padding_token: int = 0
                ends = (sample_outputs[:, start:] != padding_token).sum(1)

                # Get the logprobs and values, for tokens that are not padding
                # or beginning of sequences tokens. These are from the model
                # (not the reference model)
                all_logprobs = [
                    logprobs[ix, start : ends[ix]] for ix in range(n_samples)
                ]
                all_values = [
                    values[ix, start - 1 : ends[ix] - 1] for ix in range(n_samples)
                ]

                kl_divergence_estimate: List[torch.Tensor] = [
                    -self.kl_ctl.value
                    * (
                        logprobs[sample_idx, start : ends[sample_idx]]
                        - ref_logprobs[sample_idx, start : ends[sample_idx]]
                    )
                    for sample_idx in range(n_samples)
                ]

            # Else if not seq2seq (i.e. causal)
            else:
                values = values.cpu()[:, :-1]

                start = prompt_tensors.shape[1] - 1
                ends = start + attention_mask[:, start:].sum(1)
                all_values = [values[ix, start : ends[ix]] for ix in range(n_samples)]
                all_logprobs = [
                    logprobs[ix, start : ends[ix]] for ix in range(n_samples)
                ]

                # kl_divergence_estimate = -self.kl_ctl.value * (logprobs - ref_logprobs)
                kl_divergence_estimate = self.kl_ctl.value * -log_ratio.cpu()
                kl_divergence_estimate = [
                    rs[start : ends[ix]] for ix, rs in enumerate(kl_divergence_estimate)
                ]

            rollout_count = 0

            for sample_idx in range(n_samples):
                sample_kl_divergence_estimate = kl_divergence_estimate[sample_idx]
                rewards = sample_kl_divergence_estimate
                rewards[-1] += scores[sample_idx].cpu()
                ppo_rl_elements.append(
                    PPORLElement(
                        query_tensor=prompt_tensors[sample_idx],
                        response_tensor=sample_outputs[sample_idx],
                        logprobs=all_logprobs[sample_idx],
                        values=all_values[sample_idx],
                        rewards=rewards,
                    )
                )

                rollout_count += 1

            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    mean_kl.to(device), torch.distributed.ReduceOp.AVG
                )

            stats["time/exp"] = clock.tick()
            stats["policy/sqrt_kl"] = torch.sqrt(mean_kl).item()
            accumulate_stats.append(stats)

            tbar.set_description(f"[rollout {len(ppo_rl_elements)} / {num_rollouts}]")
            tbar.update(min(rollout_count, num_rollouts))
        tbar.close()

        stats = {
            k: sum([xs[k] for xs in accumulate_stats]) / len(accumulate_stats)
            for k in stats
        }
        stats["kl_ctl_value"] = self.kl_ctl.value
        self.mean_kl = stats["policy/sqrt_kl"] ** 2

        self.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to trainer's rollout storage
        self.push_to_store(ppo_rl_elements)
