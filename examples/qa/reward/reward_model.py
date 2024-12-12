import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class GPTRewardModel(nn.Module):
    def __init__(self, model_path, tokenizer_path="XXX", *args, **kwargs):
        super().__init__()
        self.setup_model(model_path, tokenizer_path, *args, **kwargs)

    def setup_model(self, model_path, tokenizer_path="XXX"):
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.gradient_checkpointing_enable()
        self.config = model.config
        # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
        if hasattr(self.config, "hidden_size"):
            print(f"self.config.hidden_size: {self.config.hidden_size}")

        if hasattr(self.config, "n_embd"):
            print(f"self.config.n_embd: {self.config.n_embd}")
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.model = model
        print("Model backbone:\n", model)
        self.transformer = model.model

        # Freeze the first 70% of the hidden layers of the reward model backbone
        layers = self.transformer.layers
        num_layers = len(layers)
        num_unfrozen = int(0.3 * num_layers)
        for layer in layers[:-num_unfrozen]:
            layer.requires_grad_(False)

        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if "Llama-2-" in tokenizer_path:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
                self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        elif "Llama-3-" in tokenizer_path:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.EOS_ID = self.tokenizer.eos_token_id

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        for param in self.v_head.parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        labels=None,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        loss = None
        if token_type_ids is None:
            transformer_outputs = self.transformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                # position_ids=position_ids,
                inputs_embeds=inputs_embeds,
            )
        else:
            transformer_outputs = self.transformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

        hidden_states = transformer_outputs[0]

        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_end_scores = []
        rejected_end_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        chosen = input_ids[:bs]
        rejected = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        # Compute pairwise loss. Only backprop on the last value before padding
        loss = 0
        inference = False
        for i in range(bs):
            if torch.all(torch.eq(chosen[i], rejected[i])).item():
                c_inds = (chosen[i] == self.EOS_ID).nonzero()
                c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
                chosen_end_scores.append(chosen_rewards[i, c_ind - 1])
                inference = True
                continue

            # Check if there is any padding otherwise take length of sequence
            c_inds = (chosen[i] == self.EOS_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
            r_inds = (rejected[i] == self.EOS_ID).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
            end_ind = max(c_ind, r_ind)

            try:

                # Retrieve first index where trajectories diverge
                divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]
                assert divergence_ind > 0

                # Index into the correct rewards
                c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
                r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]

                # Append the last rewards to the list of end scores
                chosen_end_scores.append(c_truncated_reward[-1])
                rejected_end_scores.append(r_truncated_reward[-1])

            except:
                torch.set_printoptions(threshold=np.inf)
                print(chosen[i])
                print("=" * 100)
                print(rejected[i])
                print("=" * 100)
                print(f"divergence ind = {divergence_ind}, end ind = {end_ind}")
                exit(0)

            # Compute loss
            loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
        loss = loss / bs

        if not inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            rejected_end_scores = torch.stack(rejected_end_scores)

        if inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            return {"chosen_end_scores": chosen_end_scores}

        return {
            "loss": loss,
            "chosen_end_scores": chosen_end_scores,
            "rejected_end_scores": rejected_end_scores,
        }


class GPTRewardModelLora(GPTRewardModel):
    def __init__(self, model_path, tokenizer_path, lora_config: LoraConfig):
        super().__init__(model_path, tokenizer_path, lora_config)

    def setup_model(self, model_path, tokenizer_path, lora_config: LoraConfig):
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model = get_peft_model(model, lora_config)

        model.gradient_checkpointing_enable()
        self.config = model.config
        # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``

        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.config.output_hidden_states = True

        self.model = model
        print("Model backbone:\n", model)

        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if "Llama-2-" in tokenizer_path:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
                self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        elif "Llama-3-" in tokenizer_path:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.EOS_ID = self.tokenizer.eos_token_id

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        labels=None,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        loss = None
        outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs.hidden_states[-1]

        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_end_scores = []
        rejected_end_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        chosen = input_ids[:bs]
        rejected = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        # Compute pairwise loss. Only backprop on the last value before padding
        loss = 0
        inference = False
        for i in range(bs):
            if torch.all(torch.eq(chosen[i], rejected[i])).item():
                c_inds = (chosen[i] == self.EOS_ID).nonzero()
                c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
                chosen_end_scores.append(chosen_rewards[i, c_ind - 1])
                inference = True
                continue

            # Check if there is any padding otherwise take length of sequence
            c_inds = (chosen[i] == self.EOS_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
            r_inds = (rejected[i] == self.EOS_ID).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
            end_ind = max(c_ind, r_ind)

            try:

                # Retrieve first index where trajectories diverge
                divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]
                assert divergence_ind > 0

                # Index into the correct rewards
                c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
                r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]

                # Append the last rewards to the list of end scores
                chosen_end_scores.append(c_truncated_reward[-1])
                rejected_end_scores.append(r_truncated_reward[-1])

            except:
                torch.set_printoptions(threshold=np.inf)
                print(chosen[i])
                print("=" * 100)
                print(rejected[i])
                print("=" * 100)
                print(f"divergence ind = {divergence_ind}, end ind = {end_ind}")
                exit(0)

            # Compute loss
            loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
        loss = loss / bs

        if not inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            rejected_end_scores = torch.stack(rejected_end_scores)

        if inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            return {"chosen_end_scores": chosen_end_scores}

        return {
            "loss": loss,
            "chosen_end_scores": chosen_end_scores,
            "rejected_end_scores": rejected_end_scores,
        }
