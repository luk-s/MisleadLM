from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    logging,
    set_seed,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

TEXT_TEMPLATE = """PARAGRAPH: {paragraph}

QUESTION: {question}

ANSWER A: {answer_a}

ANSWER B: {answer_b}

ARGUMENT: Let's think logically. {argument}

ANSWER: {answer}"""


def get_args() -> Namespace:
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
    parser.add_argument("--max_length", type=int, help="The maximum length of the input sequence.")
    parser.add_argument("--lr", type=float, help="The learning rate.")
    parser.add_argument("--batch_size", type=int, help="The batch size.")
    parser.add_argument("--max_epochs", type=int, help="The maximum number of epochs.")
    parser.add_argument("--gradient_accumulation", type=int, help="The number of gradient accumulation steps.")
    parser.add_argument("--save_steps", type=int, help="The number of steps between each save.")
    parser.add_argument("--eval_steps", type=int, help="The number of steps between each evaluation.")
    parser.add_argument(
        "--deepspeed_config", type=str, default=None, help="An optional path to a deepspeed configuration file."
    )
    return parser.parse_args()


def main(args: Namespace) -> None:
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, truncation_side="left")
    if "Llama-2-" in args.tokenizer_name:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
    elif "Llama-3-" in args.tokenizer_name:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation="flash_attention_2",
        # device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=False,  # This is required when 'gradient_checkpointing' is set to True
    )

    # Prepare the dataset
    dataset = load_dataset("json", data_files=args.train_data)["train"]

    # Remove all rows where 'choseAnswerId' is not equal to 'correctAnswerId'.
    dataset = dataset.filter(lambda example: example["choseAnswerId"] == example["correctAnswerId"])

    # Remove all rows where 'correctAnswerId' is not in [0, 1].
    dataset = dataset.filter(lambda example: example["correctAnswerId"] in [0, 1])

    # Split the dataset into a training and evaluation dataset
    train_dataset = dataset.select(range(args.num_train_samples))
    eval_dataset = dataset.select(range(args.num_train_samples, args.num_train_samples + args.num_eval_samples))

    # These steps are necessary due to some particularities of Llama2's tokenizer.
    # See: https://huggingface.co/docs/trl/main/en/sft_trainer#using-tokenids-directly-for-responsetemplate
    start_template = "PARAGRAPH: "
    response_template_with_context = "\n\nARGUMENT: Let's think logically."
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[3:8]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    def formatting_prompts_func(batch):
        output_texts = []
        for batch_index in range(len(batch["question"])):
            text = TEXT_TEMPLATE.format(
                paragraph=batch["paragraph"][batch_index],
                question=batch["question"][batch_index],
                answer_a=batch["answers"][batch_index][0],
                answer_b=batch["answers"][batch_index][1],
                argument=batch["argument"][batch_index],
                answer="A" if batch["correctAnswerId"][batch_index] == 0 else "B",
            )

            # Tokenize and detokenize the text. This truncates the text automatically to the max length
            # in case it is too long.
            tokenized_text = tokenizer.encode(text, truncation=True, max_length=args.max_length)
            detokenized_text = tokenizer.decode(tokenized_text, skip_special_tokens=True)

            # Check whether the detokenized text does start with the start template.
            # If the original text was too long, this is not the case.
            if not detokenized_text.startswith(start_template):
                # Replace the start of the detokenized text with the start template.
                # The '+50' is a buffer such that we can guarantee that the new detokenized text will be encoded to less than 'args.max_length' tokens.
                detokenized_text = start_template + detokenized_text[len(start_template) + 50 :]

                print("Detokenized text was too long. Truncated it to the max length.")

            output_texts.append(detokenized_text)
        return output_texts

    training_args = SFTConfig(
        run_name=args.run_name,
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        save_strategy="epoch" if args.save_steps is None else "steps",
        save_steps=args.save_steps,
        evaluation_strategy="epoch" if args.eval_steps is None else "steps",
        eval_steps=args.eval_steps,
        logging_steps=args.eval_steps,
        max_seq_length=args.max_length,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.max_epochs,
        fp16=False,
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
    )

    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    trainer.train()


if __name__ == "__main__":
    args = get_args()

    main(args)
