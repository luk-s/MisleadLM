from argparse import ArgumentParser, Namespace
from typing import Any, Union

import torch
from datasets import load_dataset
from qa_dataset import AGENT_SYSTEM_PROMPT, AGENT_USER_PROMPT
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

ASSISTANT_PROMPT = """<argument>{argument} Final Answer: {final_answer}</argument>"""


def get_args() -> Namespace:
    """
    Parse and return command-line arguments.

    Args:
        None

    Returns:
        Namespace: Parsed command-line arguments.
    """
    # fmt: off
    parser = ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--run_name", type=str, help="A name for the run.")
    parser.add_argument("--output_dir", type=str, help="A path to save the model and optimizer states.")
    parser.add_argument("--logging_dir", type=str, help="A path to save the logs.")
    parser.add_argument("--model_name", type=str, help="A name of- or a path to a pretrained model.")
    parser.add_argument("--tokenizer_name", type=str, help="A name of- or a path to a pretrained tokenizer.")
    parser.add_argument("--train_data", type=str, help="A path to a training dataset.")
    parser.add_argument("--num_train_samples", type=int, help="The number of training samples.")
    parser.add_argument("--num_eval_samples", type=int, help="The number of evaluation samples.")
    parser.add_argument("--lr", type=float, help="The learning rate.")
    parser.add_argument("--max_seq_length", type=int, help="The maximum sequence length.")
    parser.add_argument("--batch_size", type=int, help="The batch size.")
    parser.add_argument("--max_epochs", type=int, help="The maximum number of epochs.")
    parser.add_argument("--gradient_accumulation", type=int, help="The number of gradient accumulation steps.")
    parser.add_argument("--save_steps", type=int, help="The number of steps between each save.")
    parser.add_argument("--eval_steps", type=int, help="The number of steps between each evaluation.")
    parser.add_argument("--deepspeed_config", type=str, default=None, help="An optional path to a deepspeed configuration file.")
    # fmt: on
    return parser.parse_args()


def main(args: Namespace) -> None:
    """
    Main function to fine-tune the causal language model.

    Args:
        args (Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    # Add a padding token
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    tokenizer.pad_token_id = 128004
    tokenizer.padding_side = "right"

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation="flash_attention_2",
        # device_map="auto",
        torch_dtype=torch.bfloat16,
        # torch_dtype=torch.float16,
        use_cache=False,  # This is required when 'gradient_checkpointing' is set to True
    )

    # Prepare the dataset
    dataset = load_dataset("json", data_files=args.train_data)["train"]

    # Remove all rows where 'choseAnswerId' is not in [0, 1].
    dataset = dataset.filter(lambda example: example["choseAnswerId"] in [0, 1])

    # Split the dataset into a training and evaluation dataset
    train_dataset = dataset.select(range(args.num_train_samples))
    eval_dataset = dataset.select(
        range(args.num_train_samples, args.num_train_samples + args.num_eval_samples)
    )

    # Print the dataset sizes
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    # These steps are necessary due to some particularities of Llama's tokenizer.
    # See: https://huggingface.co/docs/trl/main/en/sft_trainer#using-tokenids-directly-for-responsetemplate
    # NOTE: This only works for Llama-3.1-8B-Instruct and needs to be updated for other models.
    # response_template_with_context = (
    #     "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    # )
    # response_template_ids = tokenizer.encode(
    #     response_template_with_context, add_special_tokens=False
    # )[2:5]
    # collator = DataCollatorForCompletionOnlyLM(
    #     response_template_ids, tokenizer=tokenizer
    # )

    # Check if the tokenizer has a chat template
    # if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
    #     response_template = "<|start_header_id|>assistant<|end_header_id|>"
    # else:
    #     response_template = "\n\nASSISTANT:"
    #
    # collator = DataCollatorForCompletionOnlyLM(
    #     response_template=response_template,
    #     tokenizer=tokenizer,
    # )

    def formatting_prompts_func(batch):
        """
        Format prompts for the training data.

        Args:
            batch (dict): A batch of data containing 'paragraph', 'question', 'answers', 'argument', and 'choseAnswerId'.

        Returns:
            list: A list of formatted text prompts.
        """
        output_texts = []
        for batch_index in range(len(batch["question"])):
            user_prompt = AGENT_USER_PROMPT.format(
                paragraph=batch["paragraph"][batch_index],
                question=batch["question"][batch_index],
                answer_a=batch["answers"][batch_index][0],
                answer_b=batch["answers"][batch_index][1],
            )
            assistant_prompt = ASSISTANT_PROMPT.format(
                argument=batch["argument"][batch_index],
                final_answer="A" if batch["choseAnswerId"][batch_index] == 0 else "B",
            )
            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_prompt},
            ]

            # Check if the tokenizer has a chat template
            if (
                hasattr(tokenizer, "chat_template")
                and tokenizer.chat_template is not None
            ):
                tokenized_message = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_bos=False
                )
            else:
                # Fallback if no chat template is available
                system_text = f"SYSTEM: {AGENT_SYSTEM_PROMPT}\n\n"
                user_text = f"USER: {user_prompt}\n\n"
                assistant_text = f"ASSISTANT: {assistant_prompt}"
                tokenized_message = system_text + user_text + assistant_text

            # For some tokenizers, we have to manually remove the BOS token because this string will be prepended with the BOS token
            # when the string is tokenized inside the 'SFTTrainer' class. This can't be done manually because there
            # only exists a FastTokenizer, so setting parameters 'add_bos=False' or 'add_special_tokens=False'
            # above in the 'tokenizer.apply_chat_template' function will just be ignored.
            # See https://github.com/huggingface/transformers/issues/30947#issuecomment-2126708114
            if tokenized_message.startswith(tokenizer.bos_token):
                tokenized_message = tokenized_message[len(tokenizer.bos_token) :]

            output_texts.append(tokenized_message)
        return output_texts

    training_args = SFTConfig(
        run_name=args.run_name,
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        save_strategy="epoch" if args.save_steps is None else "steps",
        save_steps=args.save_steps,
        eval_strategy="epoch" if args.eval_steps is None else "steps",
        eval_steps=args.eval_steps,
        logging_steps=args.eval_steps,
        max_seq_length=args.max_seq_length,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.max_epochs,
        adam_beta1=0.9,  # Must match the parameters in the deepspeed configuration file
        adam_beta2=0.95,  # Must match the parameters in the deepspeed configuration file
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,  # To save memory
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },  # Required due to this issue: https://github.com/huggingface/transformers/issues/26969
        half_precision_backend="auto",  # To use flash attention
        neftune_noise_alpha=5,  # Apparently, this increases performance quite a bit: https://huggingface.co/docs/trl/main/en/sft_trainer#enhance-the-models-performances-using-neftune
        deepspeed=args.deepspeed_config,
        metric_for_best_model="eval_loss",
        fp16=False,
        bf16=True,
        log_level="debug",
        debug="underflow_overflow",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        formatting_func=formatting_prompts_func,
        # data_collator=collator,
    )

    trainer.train()


if __name__ == "__main__":
    args = get_args()

    main(args)
