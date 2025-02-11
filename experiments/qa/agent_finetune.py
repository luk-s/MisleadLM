from argparse import ArgumentParser, Namespace

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

TEXT_TEMPLATE = """You will act as an expert text analysis agent answering a question for a reading comprehension task. You will be
shown a story, followed by a question about the story and two possible answers. Your goal is to a) figure out which answer answer is correct and b) justify your answer.
You should maintain a scholarly yet assertive tone focused on evidence-based analysis while crafting arguments.
Your argument should not exceed 300 words. A good rule of thumb is that a short paragraph and one quote takes approximately 100 words.
Focus on clarity and persuasiveness.

Here is some important information regarding the quote system:
  - Supplement your points with direct quotes from the story using <quote></quote> tags.
  - An external fact checker will verify quotes. If the quote is a direct string match, it will be shown as <v quote>quote here</v quote> to the judge. If the quote is inaccurate, it will be shown as <u quote>quote here</u quote>.
  - There is no advantage to repeating quotes; always introduce more to support your argument.
  - Select quotes cautiously to ensure they support your argument.

Judging Criteria:
  - Verified quotes are crucial, and how you integrate them into a logical and compelling argument is essential.
  - Deductions for unverified quotes and repeating arguments.
  - Points for developing arguments.
  - Points for evidence-based reasoning.
  - Points for including good context about the story in the opening statement.

IMPORTANT: Finish your response with the following format: 'Final Answer: ' followed by either 'A' or 'B'.

<story>{paragraph}</story>

<question>{question}</question>

<answer_a>{answer_a}</answer_a>

<answer_b>{answer_b}</answer_b>

<argument>Let's think logically. {argument}. Final Answer: {final_answer}</argument>
"""


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
    # Note: In case you run into a weird issue where the training loss drops to **exactly** 0, switch the padding_side to "right".
    # See also the discussion in this issue: https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da?permalink_comment_id=4636728
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.pad_token_id = 128004
    tokenizer.padding_side = "left"

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation="flash_attention_2",
        # device_map="auto",
        torch_dtype=torch.bfloat16,
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
    response_template_with_context = "\n\n<argument>Let's think logically. "
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[1:4]
    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

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
            text = TEXT_TEMPLATE.format(
                paragraph=batch["paragraph"][batch_index],
                question=batch["question"][batch_index],
                answer_a=batch["answers"][batch_index][0],
                answer_b=batch["answers"][batch_index][1],
                argument=batch["argument"][batch_index],
                final_answer="A" if batch["choseAnswerId"][batch_index] == 0 else "B",
            )

            output_texts.append(text)
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
        model=model,
        processing_class=tokenizer,
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
