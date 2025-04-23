import inspect
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from huggingface_hub import HfApi, Repository
from torchtyping import TensorType
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
)
from transformers.modeling_outputs import ModelOutput
from transformers.models.bloom import modeling_bloom
from transformers.models.opt import modeling_opt

from trlx.data.method_configs import MethodConfig, register_method
from trlx.utils.modeling import (
    flatten_dict,
    get_tensor_stats,
    hf_get_causal_base_model,
    hf_get_causal_final_norm,
    hf_get_causal_hidden_layers,
    hf_get_hidden_size,
    hf_get_lm_head,
    hf_get_num_hidden_layers,
    make_head,
    whiten,
)


def clone_hf_repo(repo_id: str, local_dir: str, repo_type: str = "model") -> None:
    """Check repo existence, clone if available, and list local files."""

    # If the 'local_dir' already exists, return
    if Path(local_dir).exists():
        print(f"Repository {repo_id} already exists in {local_dir}")
        return

    # Otherwise, clone the repo
    try:
        repo = Repository(
            local_dir=local_dir,
            clone_from=repo_id,
            repo_type=repo_type,
            skip_lfs_files=False,  # For faster cloning without LFS files
        )
        print(f"Successfully cloned to {Path(local_dir).resolve()}")
    except EnvironmentError as e:
        print(f"Cloning failed: {str(e)}")
        return


# KL Controllers
class AdaptiveKLController:
    """Adaptive KL Controller as described in Ziegler et al. "Fine-Tuning Language Models from Human Preferences"
    Reference: Section 2.2 https://arxiv.org/pdf/1909.08593.pdf#page=2
    Source: https://github.com/openai/lm-human-preferences/blob/master/lm_human_preferences/train_policy.py
    """

    def __init__(self, init_kl_coef: float, target: float, horizon: int):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current: float, n_steps: int):
        """Returns adaptively updated KL coefficient, βₜ₊₁.
        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)  # ϵₜ
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult  # βₜ₊₁


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current: float, n_steps: int):
        """Returns updated KL coefficient, βₜ₊₁.
        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
        pass


# PPO Configs


@dataclass
@register_method
class PPOConfig(MethodConfig):
    """
    Config for PPO method

    :param ppo_epochs: Number of updates per batch
    :type ppo_epochs: int

    :param num_rollouts: Number  of experiences to observe before learning
    :type num_rollouts: int

    :param init_kl_coef: Initial value for KL coefficient
    :type init_kl_coef: float

    :param target: Target value for KL coefficient
    :type target: float

    :param horizon: Number of steps for KL coefficient to reach target
    :type horizon: int

    :param gamma: Discount factor
    :type gamma: float

    :param lam: GAE lambda
    :type lam: float

    :param cliprange: Clipping range for PPO policy loss (1 - cliprange, 1 + cliprange)
    :type cliprange: float

    :param cliprange_value: Clipping range for predicted values
                            (observed values - cliprange_value, observed values + cliprange_value)
    :type cliprange_value: float

    :param vf_coef: Value loss scale w.r.t policy loss
    :type vf_coef: float

    :param gen_kwargs: Additioanl kwargs for the generation
    :type gen_kwargs: Dict[str, Any]

    :param gen_experience_kwargs: if this is not None, then the experience is generated using this
    :type gen_experience_kwargs: Dict[str, Any]
    """

    ppo_epochs: int
    num_rollouts: int
    chunk_size: int
    init_kl_coef: float
    target: float
    horizon: int
    gamma: float
    lam: float
    cliprange: float
    cliprange_value: float
    vf_coef: float
    scale_reward: str
    ref_mean: Optional[float]
    ref_std: Optional[float]
    cliprange_reward: float
    gen_kwargs: dict
    gen_experience_kwargs: Optional[dict] = None

    def get_advantages_and_returns(
        self,
        values: TensorType["batch_size", "response_size"],
        rewards: TensorType["batch_size", "response_size"],
        response_length: int,
        use_whitening: Optional[bool] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)
        - response_length: Length of the response sequence
        - use_whitening: Whether to use whitening (ie. normalize advantages) or not
        """
        lastgaelam = 0
        advantages_reversed = []
        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        if use_whitening:
            advantages = whiten(advantages)
        return advantages.detach(), returns

    def loss(
        self,
        logprobs: TensorType["batch_size", "response_size"],
        values: TensorType["batch_size", "response_size"],
        old_logprobs: TensorType["batch_size", "response_size"],
        old_values: TensorType["batch_size", "response_size"],
        advantages: TensorType["batch_size", "response_size"],
        returns: TensorType["batch_size", "response_size"],
        mask: TensorType["batch_size", "response_size"],
    ):
        """PPO objective function.
        References:
        - https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
        """
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        n = mask.sum()

        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / n
        vf_clipfrac = torch.sum((vf_loss2 > vf_loss1).float() * mask) / n

        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        # Unbiased KL-div estimates (`k3`). Ref: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
            approx_kl = torch.mean(torch.sum((ratio - 1) - log_ratio, dim=-1))

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.cliprange,
            1.0 + self.cliprange,
        )
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / n
        pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * mask) / n

        loss = pg_loss + self.vf_coef * vf_loss

        stats = dict(
            losses=dict(
                total_loss=loss.item(),
                policy_loss=pg_loss.item(),
                value_loss=vf_loss.item(),
            ),
            values=dict(
                get_tensor_stats(values, mask, n),
                values_error=torch.sum(((values - returns) * mask) ** 2) / n,
                clipfrac=vf_clipfrac,
            ),
            old_values=get_tensor_stats(old_values, mask, n),
            returns=get_tensor_stats(returns, mask, n),
            policy=dict(approx_kl=approx_kl.item(), clipfrac=pg_clipfrac.item()),
            ratio=(ratio * mask).sum() / n,
            padding_percentage=n / mask.numel(),
        )

        return loss, flatten_dict(stats)


# PPO Layers


@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    value: Optional[torch.FloatTensor] = None


class CausalLMWithValueHead(nn.Module):
    """The CausalLMWithValueModel class implements a causal language model with
    a secondary, scalar head.
    """

    def __init__(self, config: Union[transformers.PretrainedConfig, str]):
        super().__init__()
        if isinstance(config, str):
            self.config = transformers.AutoConfig.from_pretrained(config)
            self.base_model = transformers.AutoModelForCausalLM.from_pretrained(config)
        else:
            self.config = config
            self.base_model = transformers.AutoModelForCausalLM.from_config(config)

        self.base_model.transformer = hf_get_causal_base_model(self.base_model)
        self.base_model.lm_head = hf_get_lm_head(self.base_model)
        dtype = next(self.base_model.lm_head.parameters()).dtype
        self.v_head = make_head(hf_get_hidden_size(self.config), 1, dtype)

        # Cache `transformer.forward` args for general use (avoids incompatible args across architectures)
        self.base_model_transformer_args = inspect.getfullargspec(
            self.base_model.transformer.forward
        ).args

    def _get_compatible_forward_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Filter out arguments not supported by the specific instance of `base_model.transformer.forward`"""
        return {
            k: v for k, v in kwargs.items() if k in self.base_model_transformer_args
        }

    def generate(self, input_ids, **kwargs):
        return self.base_model.generate(input_ids, **kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        return_dict=False,
    ):
        forward_kwargs = self._get_compatible_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        transformer_outputs = self.base_model.transformer(**forward_kwargs)
        last_hidden_state = transformer_outputs.last_hidden_state
        lm_logits = self.base_model.lm_head(last_hidden_state)
        value = self.v_head(last_hidden_state).squeeze(-1)

        if not return_dict:
            outputs = (lm_logits,) + transformer_outputs[1:] + (value,)
            return outputs

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            value=value,
        )


class CausalLMHydraWithValueHead(nn.Module):
    """The CausalLMHydraWithValueHead class implements a causal language model
    with a secondary, scalar head.
    """

    def __init__(
        self,
        config: Union[transformers.PretrainedConfig, str],
        num_layers_unfrozen: int = -1,
        load_weights: bool = True,
        build_complete_model: bool = True,
    ):
        super().__init__()
        print("config = ", config)

        self.num_layers_unfrozen = num_layers_unfrozen

        self.setup_base_model(config, load_weights)

        if build_complete_model:
            self.build_complete_model()

    def setup_base_model(
        self, config: Union[transformers.PretrainedConfig, str], load_weights=True
    ):
        # from pre_trained
        if isinstance(config, str):
            if "glm" in config:
                self.config = transformers.AutoConfig.from_pretrained(
                    config, trust_remote_code=True
                )
                self.base_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                    config, trust_remote_code=True
                )
            else:
                print("gradient checkpointing & bf16")
                self.config = transformers.AutoConfig.from_pretrained(
                    config, trust_remote_code=True
                )
                if load_weights:
                    self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
                        config, trust_remote_code=True
                    )
                else:
                    self.base_model = transformers.AutoModelForCausalLM.from_config(
                        self.config
                    )
        else:
            self.config = config
            self.base_model = transformers.AutoModelForCausalLM.from_config(config)

    def build_complete_model(self):
        self.base_model.transformer = hf_get_causal_base_model(self.base_model)
        self.base_model.lm_head = hf_get_lm_head(self.base_model)
        dtype = next(self.base_model.lm_head.parameters()).dtype
        self.v_head = make_head(hf_get_hidden_size(self.config), 1, dtype)

        if self.num_layers_unfrozen > 0:
            transformer_blocks = list(hf_get_causal_hidden_layers(self.base_model))
            branch_class = hf_get_causal_lm_branch_class(self.config)
            if branch_class == GLMModelBranch:
                self.frozen_head = branch_class(
                    self.config,
                    transformer_blocks[-self.num_layers_unfrozen :],
                    final_norm=hf_get_causal_final_norm(self.base_model),
                    lm_head=self.base_model.lm_head,
                    embedding_dropout=self.base_model.transformer.transformer.embedding_dropout,
                    position_embeddings=self.base_model.transformer.transformer.position_embeddings,
                    block_position_embeddings=self.base_model.transformer.transformer.block_position_embeddings,
                )
            else:
                self.frozen_head = branch_class(
                    self.config,
                    transformer_blocks[-self.num_layers_unfrozen :],
                    final_norm=hf_get_causal_final_norm(self.base_model),
                    lm_head=self.base_model.lm_head,
                )
        else:
            # This is required because some parts of the 'trlx' library will check whether the frozen head is present
            self.frozen_head = "This is a dummy frozen head"

        # Cache `transformer.forward` args for general use (avoids incompatible args across architectures)
        self.base_model_transformer_args = inspect.getfullargspec(
            self.base_model.transformer.forward
        ).args

    def _get_compatible_forward_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Filter out arguments not supported by the specific instance of `base_model.transformer.forward`"""
        return {
            k: v for k, v in kwargs.items() if k in self.base_model_transformer_args
        }

    def generate(self, input_ids, **x):
        return self.base_model.generate(input_ids, **x)

    def forward_hydra(self, input_ids, **forward_kwargs):
        forward_kwargs = self._get_compatible_forward_kwargs(**forward_kwargs)
        if forward_kwargs.get("return_dict") is not None:
            return_dict = forward_kwargs["return_dict"]
        else:
            return_dict = True
        forward_kwargs["return_dict"] = True
        forward_kwargs["output_hidden_states"] = True
        output = self.forward(input_ids, **forward_kwargs)
        all_hidden_states = output.hidden_states
        # Get output of last frozen hidden layer
        # Select hidden state before first layer of branch.
        input_hidden_state = all_hidden_states[-(self.num_layers_unfrozen + 1)]
        # Get size of last hidden state
        output_shape = all_hidden_states[-1].size()
        outputs = self.frozen_head(input_hidden_state, output_shape, **forward_kwargs)
        if not return_dict:
            return outputs.logits
        return outputs

    def state_dict(self, *args, heads_only=False, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        state_dict = self.v_head.state_dict(*args, **dict(prefix="v_head.", **kwargs))
        if not heads_only:
            state_dict = {
                **state_dict,
                **self.base_model.state_dict(
                    *args, **dict(prefix="base_model.", **kwargs)
                ),
            }

            if self.frozen_head:
                state_dict = {
                    **state_dict,
                    **self.frozen_head.state_dict(
                        *args, **dict(prefix="frozen_head.", **kwargs)
                    ),
                }

        return state_dict

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
    ):
        forward_kwargs = self._get_compatible_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            token_type_ids=token_type_ids,
        )
        transformer_outputs = self.base_model.transformer(**forward_kwargs)
        # print('transformer outputs.keys = ', transformer_outputs.keys())
        last_hidden_state = transformer_outputs.last_hidden_state
        # print('last hidden state.shape = ', last_hidden_state.shape)
        if self.config.architectures[0] == "GLMModel":  # glm
            lm_logits = transformer_outputs.logits
            hidden_states = transformer_outputs.mems
            # print('hidden states len = ', len(hidden_states))
            # print('shape = ', hidden_states[0].shape)
        else:
            lm_logits = self.base_model.lm_head(last_hidden_state)
            hidden_states = transformer_outputs.hidden_states
        value = self.v_head(last_hidden_state).squeeze(-1)

        if not return_dict:
            outputs = (lm_logits,) + transformer_outputs[1:] + (value,)
            return outputs

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            # past_key_values=transformer_outputs.past_key_values,
            hidden_states=hidden_states,
            # attentions=transformer_outputs.attentions,
            cross_attentions=None,
            value=value,
        )

    def save_pretrained(self, *args, **kwargs):
        """Save the pretrained model to a directory. This method is a wrapper
        around `transformers.PreTrainedModel.save_pretrained`. Please refer to
        the documentation of `transformers.PreTrainedModel.save_pretrained` for
        more information.

        Args:
            *args (`list`, *optional*):
                Positional arguments passed along to the underlying model's
                `save_pretrained` method.
            **kwargs (`dict`, *optional*):
                Keyword arguments passed along to the underlying model's
                `save_pretrained` method.
        """
        state_dict = kwargs.get("state_dict", None)
        if state_dict is None:
            state_dict = self.state_dict()
            kwargs["state_dict"] = state_dict
        return self.base_model.save_pretrained(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        config,
        num_layers_unfrozen=-1,
        hf_local_target_dir=None,
        **kwargs,
    ):
        """
        Loads a CausalLMHydraWithValueHead model from a pretrained checkpoint.

        Args:
            pretrained_model_name_or_path (str): Name of hf repo or path to the pretrained checkpoint.
            config (Union[transformers.PretrainedConfig, str]): Model configuration.
            num_layers_unfrozen (int, optional): Number of layers to unfreeze. Defaults to -1.
            hf_local_target_dir (str, optional): Target directory for the cloned hf repo. Defaults to None.
            **kwargs: Additional arguments.

        Returns:
            CausalLMHydraWithValueHead: The loaded model.
        """

        # Instantiate an empty shell of the base model
        model = cls(
            config,
            num_layers_unfrozen=num_layers_unfrozen,
            build_complete_model=False,
            load_weights=False,
            **kwargs,
        )

        # Check whether pretrained_model_name_or_path is a hf repo
        hf_api = HfApi()
        if hf_api.repo_exists(repo_id=pretrained_model_name_or_path, repo_type="model"):
            assert hf_local_target_dir is not None, (
                "'hf_local_target_dir' must be provided if pretrained_model_name_or_path is a huggingface repo"
            )

            # Clone the repo
            clone_hf_repo(
                repo_id=pretrained_model_name_or_path, local_dir=hf_local_target_dir
            )
            pretrained_model_name_or_path = hf_local_target_dir

        else:
            if hf_local_target_dir is not None:
                print(
                    "Warning: pretrained_model_name_or_path is not a huggingface repo, but 'hf_local_target_dir' was provided. This will be ignored."
                )

        # Load state_dict, handling both single and sharded checkpoints
        checkpoint_path = Path(pretrained_model_name_or_path)
        if checkpoint_path.is_dir():
            # Find all shard files matching the pattern 'pytorch_model-*-of-*.bin'
            shard_pattern1 = checkpoint_path / "pytorch_model-*-of-*.bin"
            shard_pattern2 = checkpoint_path / "model-*-of-*.safetensors"
            shard_files1 = sorted(shard_pattern1.parent.glob(shard_pattern1.name))
            shard_files2 = sorted(shard_pattern2.parent.glob(shard_pattern2.name))

            if not shard_files1 and not shard_files2:
                raise ValueError(
                    f"No shard files found in directory: {checkpoint_path}"
                )

            state_dict = {}
            for shard_file in shard_files1:
                print(f"Loading shard: {shard_file}")
                shard_state = torch.load(
                    shard_file, map_location="cpu", weights_only=True
                )
                state_dict.update(shard_state)
            for shard_file in shard_files2:
                print(f"Loading shard: {shard_file}")
                shard_state = torch.load(
                    shard_file, map_location="cpu", weights_only=True
                )
                state_dict.update(shard_state)
        else:
            state_dict = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )

        # Separate state_dict into components
        base_model_model_state_dict = {}
        base_model_rest_state_dict = {}
        v_head_state_dict = {}
        frozen_head_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("v_head."):
                new_key = key.replace("v_head.", "")
                v_head_state_dict[new_key] = value
            elif (
                key.startswith("base_model.")
                or key.startswith("model.")
                or key.startswith("lm_head.")
            ):
                new_key = key.replace("base_model.", "")
                if new_key.startswith("model") or new_key.startswith("lm_head"):
                    base_model_model_state_dict[new_key] = value
                else:
                    base_model_rest_state_dict[new_key] = value
            elif (
                key.startswith("frozen_head.")
                and hasattr(model, "frozen_head")
                and model.frozen_head is not None
            ):
                print(f"Adding key:{key} to frozen_head_state_dict")
                new_key = key.replace("frozen_head.", "")
                frozen_head_state_dict[new_key] = value

        # Print the keys of all the state_dicts
        # print(f'base_model_model_state_dict keys:\n{base_model_model_state_dict.keys()}\n\n')
        # print(f'base_model_rest_state_dict keys:\n{base_model_rest_state_dict.keys()}\n\n')
        # print(f'v_head_state_dict keys:\n{v_head_state_dict.keys()}\n\n')
        # print(f'frozen_head_state_dict keys:\n{frozen_head_state_dict.keys()}\n\n')
        # print(f"Base model structure: {model.base_model}")

        # Populate the base model with the weights
        assert base_model_model_state_dict, "base_model_model_state_dict is empty"

        if not base_model_rest_state_dict:
            print(
                "Only model weights found, building complete model from these weights"
            )
            # We only have weights for the model part of the base model, so first load the weight of the model part, then build the complete model from these weights
            model.base_model.load_state_dict(base_model_model_state_dict, strict=True)
            model.build_complete_model()

        else:
            print("Full base model weights found, loading these weights")
            # We do have weights for the entire base model, so first build the complete base model shell, then load the weights
            model.build_complete_model()
            base_model_state_dict = {
                **base_model_model_state_dict,
                **base_model_rest_state_dict,
            }
            model.base_model.load_state_dict(base_model_state_dict, strict=True)

        # Optional: Populate the remaining components with the weights
        if v_head_state_dict:
            model.v_head.load_state_dict(v_head_state_dict, strict=True)

        if frozen_head_state_dict:
            if hasattr(model, "frozen_head") and model.frozen_head is not None:
                model.frozen_head.load_state_dict(frozen_head_state_dict, strict=True)
            else:
                print(
                    "Warning: State dict contains frozen_head keys but the model does not have a frozen_head."
                )
        else:
            print("Warning: State dict does not contain frozen_head keys.")

        return model


@dataclass
class Seq2SeqLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    value: Optional[torch.FloatTensor] = None


class Seq2SeqLMHydraWithValueHead(nn.Module):
    def __init__(
        self,
        config: Union[transformers.PretrainedConfig, str],
        num_layers_unfrozen: int = -1,
    ):
        super().__init__()
        if isinstance(config, str):
            self.config = transformers.AutoConfig.from_pretrained(config)
        else:
            self.config = config
        self.base_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            self.config.name_or_path
        )
        self.v_head = make_head(hf_get_hidden_size(self.config), 1)

        self.num_layers_unfrozen = num_layers_unfrozen
        if self.num_layers_unfrozen > 0:
            self.frozen_head = T5Branch(
                self.config, self.base_model, self.num_layers_unfrozen
            )
        # Cache `transformer.forward` args for general use (avoids incompatible args across architectures)
        self.base_model_args = inspect.getfullargspec(self.base_model.forward).args

    def _get_compatible_forward_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Filter out arguments not supported by the specific instance of `base_model.transformer.forward`"""
        return {k: v for k, v in kwargs.items() if k in self.base_model_args}

    def generate(self, input_ids, **x):
        return self.base_model.generate(input_ids, **x)

    def forward_hydra(
        self, input_ids, attention_mask, decoder_input_ids, **forward_kwargs
    ):
        forward_kwargs = self._get_compatible_forward_kwargs(**forward_kwargs)
        forward_kwargs["return_dict"] = True
        output = self.forward(
            input_ids, attention_mask, decoder_input_ids, **forward_kwargs
        )
        all_hidden_states = output.decoder_hidden_states
        # Get output of last frozen hidden layer
        # Select hidden state before first layer of branch.
        input_hidden_state = all_hidden_states[-(self.num_layers_unfrozen + 1)]
        encoder_hidden_states = output.encoder_last_hidden_state
        # Get size of last hidden state
        outputs = self.frozen_head(
            decoder_input_ids,
            input_hidden_state,
            encoder_hidden_states,
            attention_mask,
            False,
            False,
        )
        return outputs.logits

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
    ):
        forward_kwargs = self._get_compatible_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        t5_outputs = self.base_model(**forward_kwargs)
        lm_logits = t5_outputs.logits
        last_hidden_state = t5_outputs.decoder_hidden_states[-1]
        value = self.v_head(last_hidden_state).squeeze(-1)

        return Seq2SeqLMOutput(
            loss=None,
            logits=lm_logits,
            decoder_hidden_states=t5_outputs.decoder_hidden_states,
            decoder_attentions=t5_outputs.decoder_attentions,
            cross_attentions=t5_outputs.cross_attentions,
            encoder_last_hidden_state=t5_outputs.encoder_last_hidden_state,
            encoder_hidden_states=t5_outputs.encoder_hidden_states,
            encoder_attentions=t5_outputs.encoder_attentions,
            past_key_values=t5_outputs.past_key_values,
            value=value,
        )


class T5Branch(transformers.PreTrainedModel):
    # Decoder branch only
    def __init__(
        self,
        config: transformers.PretrainedConfig,
        base_model: transformers.PreTrainedModel,
        num_layers_unfrozen: int,
    ):
        super().__init__(config)

        # Defined by the main trunk
        self.hidden_size = hf_get_hidden_size(config)
        self.decoder = deepcopy(base_model.decoder)
        self.decoder.block = nn.ModuleList(self.decoder.block[-num_layers_unfrozen:])
        self.lm_head = deepcopy(base_model.lm_head)
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.last_device = None
        self.gradient_checkpointing = False

        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        input_ids,
        hidden_states,
        encoder_hidden_states,
        encoder_attention_mask,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape

        attention_mask = torch.ones(batch_size, seq_length, device=hidden_states.device)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()

        encoder_extended_attention_mask = self.invert_attention_mask(
            encoder_attention_mask
        )
        position_bias = None
        encoder_decoder_position_bias = None

        for i, layer_module in enumerate(self.decoder.block):
            layer_outputs = layer_module(
                hidden_states,  # size: (batch_size, seq_length, hidden_size)
                attention_mask=extended_attention_mask,  # size: (batch_size, 1, seq_length, seq_length)
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states

        hidden_states = self.decoder.final_layer_norm(hidden_states)
        hidden_states = self.decoder.dropout(hidden_states)
        lm_logits = self.lm_head(hidden_states)

        return Seq2SeqLMOutput(logits=lm_logits)


class GPTModelBranch(transformers.PreTrainedModel):
    """
    GPTModelBranch implements the frozen upper trunk of the reference model
    used when computing the PPO KL-divergence penalty. Expects a list of
    frozen transformer blocks and an lm_head from the base model.
    """

    def __init__(
        self,
        config: transformers.PretrainedConfig,
        transformer_blocks: nn.ModuleList,
        final_norm: nn.Module,
        lm_head: nn.Module,
    ):
        super().__init__(config)

        # Defined by the main trunk
        self.hidden_size = hf_get_hidden_size(config)
        self.transformer_blocks = deepcopy(nn.ModuleList(transformer_blocks))
        self.final_norm = deepcopy(final_norm)
        self.lm_head = deepcopy(lm_head)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Turning off grad saves memory

        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(  # noqa: max-complexity
        self,
        hidden_states: torch.Tensor,  # Takes as input hidden_states instead of input_ids
        output_shape: torch.Tensor,  # output_size given by main trunk
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        batch_size = hidden_states.size()[0]

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        device = hidden_states.device

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.transformer_blocks))

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, hf_get_num_hidden_layers(self.config))

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(
            zip(self.transformer_blocks, past_key_values)
        ):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(
                        past_state.to(hidden_states.device) for past_state in layer_past
                    )
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Assumes we are never training the branch
            block_params = inspect.getfullargspec(block.forward).args
            if "encoder_hidden_states" in block_params:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (
                        outputs[3 if use_cache else 2],
                    )

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_norm(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # last_hidden_state = hidden_states
        # past_key_values = presents
        # hidden_states = all_hidden_states
        # attentions = all_self_attentions
        # cross_attentions = all_cross_attentions

        # START OF CAUSAL HEAD #
        # hidden_states = hidden_states.to(torch.float32) Present for gptj

        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            outputs = (lm_logits,) + (None,) + (None,)
            return outputs

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
            value=None,
        )


class GLMModelBranch(transformers.PreTrainedModel):
    def __init__(
        self,
        config: transformers.PretrainedConfig,
        transformer_blocks: nn.ModuleList,
        final_norm: nn.Module,
        lm_head: nn.Module,
        embedding_dropout,
        position_embeddings,
        block_position_embeddings,
    ):
        super().__init__(config)

        # Defined by the main trunk
        self.hidden_size = hf_get_hidden_size(config)
        self.transformer_blocks = deepcopy(nn.ModuleList(transformer_blocks))
        self.final_norm = deepcopy(final_norm)
        self.lm_head = deepcopy(lm_head)
        self.position_embeddings = deepcopy(position_embeddings)
        self.block_position_embeddings = deepcopy(block_position_embeddings)

        self.embedding_dropout = deepcopy(embedding_dropout)
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Turning off grad saves memory

        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(  # noqa: max-complexity
        self,
        hidden_states: torch.Tensor,  # Takes as input hidden_states instead of input_ids
        output_shape: torch.Tensor,  # output_size given by main trunk
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        batch_size, query_length = hidden_states.size()[:2]
        memory_length = 0
        is_scalar = torch.numel(attention_mask) == 1
        is_sep = is_scalar or torch.numel(attention_mask) == batch_size
        if is_sep:
            sep = attention_mask.item() if is_scalar else attention_mask

            # conventional transformer
            def build_mask_matrix(seq_length, sep, memory_length=0):
                m = hidden_states.new_ones((1, seq_length, seq_length))
                m = torch.tril(m)
                if is_scalar:
                    m[0, :, : int(sep)] = 1
                else:
                    m = m.expand(batch_size, -1, -1)
                    ids = torch.arange(
                        seq_length, device=sep.device, dtype=sep.dtype
                    ).view(1, -1)
                    mask = ids < sep.view(-1, 1)
                    m = m.masked_fill(mask.unsqueeze(1).expand_as(m), 1)
                if memory_length > 0:
                    m = m.expand(batch_size, -1, -1)
                    m = torch.cat(
                        (
                            hidden_states.new_ones(
                                (batch_size, seq_length, memory_length)
                            ),
                            m,
                        ),
                        dim=2,
                    )
                m = m.unsqueeze(1)
                return m

            attention_mask = build_mask_matrix(
                query_length, sep, memory_length=memory_length
            )
        else:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask[:, :, :, -query_length - memory_length :]

        # position_embedding已经加完了
        # if self.block_position_encoding:
        # 默认开启block position encoding
        # position_ids, block_position_ids = position_ids[:, 0], position_ids[:, 1]
        # position_embeddings = self.position_embeddings(position_ids)

        # hidden_states = hidden_states + position_embeddings
        # if self.block_position_encoding:
        # block_position_embeddings = self.block_position_embeddings(block_position_ids)
        # hidden_states = hidden_states + block_position_embeddings
        # hidden_states = self.embedding_dropout(hidden_states)

        # def check_detach(_hidden_states):
        #     return _hidden_states.detach()

        all_hidden_states = () if output_hidden_states else None

        for i, layer in enumerate(self.transformer_blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            args = [hidden_states, attention_mask]
            mem_i = None
            hidden_states = layer(*args, mem=mem_i)

        # Final layer norm.
        output = self.final_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (output,)
        lm_logits = F.linear(output, self.lm_head.weight)

        return CausalLMOutputWithCrossAttentions(
            logits=lm_logits, hidden_states=all_hidden_states
        )


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full(
        (tgt_len, tgt_len),
        torch.tensor(torch.finfo(dtype).min, device=device),
        device=device,
    )
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


class LlamaModelBranch(transformers.PreTrainedModel):
    """
    LlamaModelBranch implements the frozen upper trunk of the reference model
    used when computing the PPO KL-divergence penalty. Expects a list of
    frozen transformer blocks and an lm_head from the base model.
    """

    def __init__(
        self,
        config: transformers.PretrainedConfig,
        transformer_blocks: nn.ModuleList,
        final_norm: nn.Module,
        lm_head: nn.Module,
    ):
        super().__init__(config)

        # Defined by the main trunk
        self.hidden_size = hf_get_hidden_size(config)
        self.transformer_blocks = deepcopy(nn.ModuleList(transformer_blocks))
        self.final_norm = deepcopy(final_norm)
        self.lm_head = deepcopy(lm_head)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Turning off grad saves memory

        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(  # noqa: max-complexity
        self,
        hidden_states: torch.Tensor,  # Takes as input hidden_states instead of input_ids
        output_shape: torch.Tensor,  # output_size given by main trunk
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        batch_size = hidden_states.size()[0]

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        batch_size, seq_length, _ = hidden_states.shape
        device = hidden_states.device

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            hidden_states,
            past_key_values_length,
        )

        # decoder layers
        # all_hidden_states = () if output_hidden_states else None
        # all_self_attentions = () if output_attentions else None
        # next_decoder_cache = () if use_cache else None

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )
        all_hidden_states = () if output_hidden_states else None
        for i, block in enumerate(self.transformer_blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (
                        outputs[3 if use_cache else 2],
                    )

        hidden_states = self.final_norm(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            outputs = (lm_logits,) + (None,) + (None,)
            return outputs

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=None,
            value=None,
        )


class DeepseekV2Branch(transformers.PreTrainedModel):
    """
    LlamaModelBranch implements the frozen upper trunk of the reference model
    used when computing the PPO KL-divergence penalty. Expects a list of
    frozen transformer blocks and an lm_head from the base model.
    """

    def __init__(
        self,
        config: transformers.PretrainedConfig,
        transformer_blocks: nn.ModuleList,
        final_norm: nn.Module,
        lm_head: nn.Module,
    ):
        super().__init__(config)

        # Defined by the main trunk
        self.hidden_size = hf_get_hidden_size(config)
        self.transformer_blocks = deepcopy(nn.ModuleList(transformer_blocks))
        self.final_norm = deepcopy(final_norm)
        self.lm_head = deepcopy(lm_head)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        self._use_flash_attention_2 = False

        # Turning off grad saves memory
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        hidden_states: torch.Tensor,  # Takes as input hidden_states instead of input_ids
        output_shape: torch.Tensor,  # output_size given by main trunk
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        batch_size, seq_length = hidden_states.shape[:2]

        past_key_values_length = 0
        # if use_cache:
        # use_legacy_cache = not isinstance(past_key_values, Cache)
        # if use_legacy_cache:
        #     past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        # past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                hidden_states,
                past_key_values_length,
            )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for i, block in enumerate(self.transformer_blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.final_norm(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            outputs = (lm_logits,) + (None,) + (None,)
            return outputs

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=None,
            value=None,
        )

    def _forward(  # noqa: max-complexity
        self,
        hidden_states: torch.Tensor,  # Takes as input hidden_states instead of input_ids
        output_shape: torch.Tensor,  # output_size given by main trunk
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        batch_size = hidden_states.size()[0]

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        batch_size, seq_length, _ = hidden_states.shape
        device = hidden_states.device

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            hidden_states,
            past_key_values_length,
        )

        # decoder layers
        # all_hidden_states = () if output_hidden_states else None
        # all_self_attentions = () if output_attentions else None
        # next_decoder_cache = () if use_cache else None

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )
        all_hidden_states = () if output_hidden_states else None
        for i, block in enumerate(self.transformer_blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (
                        outputs[3 if use_cache else 2],
                    )

        hidden_states = self.final_norm(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            outputs = (lm_logits,) + (None,) + (None,)
            return outputs

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=None,
            value=None,
        )


class OPTModelBranch(transformers.PreTrainedModel):
    """
    OPTModelBranch implements the frozen upper trunk of the reference model
    used when computing the PPO KL-divergence penalty. Expects a list of
    frozen transformer blocks and an lm_head from the base model.
    """

    def __init__(
        self,
        config: transformers.PretrainedConfig,
        transformer_blocks: nn.ModuleList,
        final_norm: nn.Module,
        lm_head: nn.Module,
    ):
        super().__init__(config)

        # Defined by the main trunk
        self.hidden_size = hf_get_hidden_size(config)
        self.transformer_blocks = deepcopy(nn.ModuleList(transformer_blocks))
        self.final_norm = deepcopy(final_norm)
        self.lm_head = deepcopy(lm_head)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Turning off grad saves memory
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(  # noqa: max-complexity
        self,
        hidden_states: torch.Tensor,  # Takes as input hidden_states instead of input_ids
        output_shape: torch.Tensor,  # output_size given by main trunk
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        """Override OPTForCausalLM"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        #######################################################################
        # Modififed OPTDecoder.forward
        #######################################################################

        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
            )

        input_shape = hidden_states.size()[:-1]
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = modeling_opt._make_causal_mask(
                input_shape,
                hidden_states.dtype,
                hidden_states.device,
                past_key_values_length=past_key_values_length,
            ).to(hidden_states.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = modeling_opt._expand_mask(
                attention_mask, hidden_states.dtype, tgt_len=input_shape[-1]
            ).to(hidden_states.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )
        attention_mask = combined_attention_mask

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.transformer_blocks)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.transformer_blocks)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.transformer_blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            layer_outputs = decoder_layer(
                hidden_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)

        # TODO: Add output projection support
        # https://github.com/huggingface/transformers/blob/699e90437f984d69ad3c9b891dd2e9d0fc2cffe4/src/transformers/models/opt/modeling_opt.py#L499  # noqa: E501
        # if self.project_out is not None:
        #     hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        #######################################################################
        # End of modified OPTDecoder.forward
        #######################################################################

        lm_logits = self.lm_head(hidden_states).contiguous()

        if not return_dict:
            return tuple(
                v
                for v in [
                    lm_logits,
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=None,
            value=None,
        )


class BloomModelBranch(transformers.PreTrainedModel):
    """
    BloomModelBranch implements the frozen upper trunk of the reference model
    used when computing the PPO KL-divergence penalty. Expects a list of
    frozen transformer blocks and an lm_head from the base model.
    """

    def __init__(
        self,
        config: transformers.PretrainedConfig,
        transformer_blocks: nn.ModuleList,
        final_norm: nn.Module,
        lm_head: nn.Module,
    ):
        super().__init__(config)

        # Defined by the main trunk
        self.hidden_size = hf_get_hidden_size(config)
        self.transformer_blocks = deepcopy(nn.ModuleList(transformer_blocks))
        self.final_norm = deepcopy(final_norm)
        self.lm_head = deepcopy(lm_head)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Turning off grad saves memory
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(  # noqa: C901
        self,
        hidden_states: torch.Tensor,  # Takes as input hidden_states instead of input_ids
        output_shape: torch.Tensor,  # output_size given by main trunk
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        #######################################################################
        # Modififed BloomModel.forward
        #######################################################################

        batch_size, seq_length = hidden_states.shape[:2]

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.transformer_blocks))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, hf_get_num_hidden_layers(self.config))

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check modeling_bloom.build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), device=hidden_states.device
            )
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = modeling_bloom.build_alibi_tensor(
            attention_mask, self.config.n_head, dtype=hidden_states.dtype
        )

        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        device = attention_mask.device
        input_shape = (batch_size, seq_length)
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = modeling_bloom._make_causal_mask(
                input_shape,
                device=device,
                past_key_values_length=past_key_values_length,
            )

        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = modeling_bloom._expand_mask(
            attention_mask, tgt_length=src_length
        )
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask | combined_attention_mask
        )
        causal_mask = combined_attention_mask

        for i, (block, layer_past) in enumerate(
            zip(self.transformer_blocks, past_key_values)
        ):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=causal_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                alibi=alibi,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )

        # Add last hidden state
        hidden_states = self.final_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        #######################################################################
        # End of modified BloomModel.forward
        #######################################################################

        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return tuple(
                v
                for v in [
                    lm_logits,
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=None,
            value=None,
        )


def hf_get_causal_lm_branch_class(
    config: transformers.PretrainedConfig,
) -> "ModelBranch":
    """Returns the CausalLM branch class for the given config."""
    gpt_branch_supported_archs = [
        "GPTJForCausalLM",
        "GPT2LMHeadModel",
        "GPTNeoForCausalLM",
        "GPTNeoXForCausalLM",
    ]
    glm_branch_supported_archs = ["GLMModel"]
    llama_branch_supported_archs = ["LlamaForCausalLM"]
    deepseek_branch_supported_archs = ["DeepseekV2ForCausalLM"]
    opt_branch_supported_archs = ["OPTForCausalLM"]
    bloom_branch_supported_archs = ["BloomModel", "BloomForCausalLM"]
    arch = config.architectures[0]
    if arch in gpt_branch_supported_archs:
        return GPTModelBranch
    elif arch in llama_branch_supported_archs:
        return LlamaModelBranch
    elif arch in opt_branch_supported_archs:
        return OPTModelBranch
    elif arch in bloom_branch_supported_archs:
        return BloomModelBranch
    elif arch in glm_branch_supported_archs:
        return GLMModelBranch
    elif arch in deepseek_branch_supported_archs:
        return DeepseekV2Branch
    else:
        all_supported_archs = sum(
            [
                gpt_branch_supported_archs,
                opt_branch_supported_archs,
                bloom_branch_supported_archs,
            ],
            [],
        )
        raise ValueError(
            f"Unsupported architecture: `{arch}`. The following architectures are "
            f"available for model branching:\n{all_supported_archs}"
        )
