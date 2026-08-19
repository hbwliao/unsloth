import argparse
import os

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported


def parse_args():
    parser = argparse.ArgumentParser(description="Merge an Unsloth LoRA checkpoint into a standalone model.")
    parser.add_argument("--checkpoint", required=True, help="Path to the LoRA checkpoint directory.")
    parser.add_argument("--output", required=True, help="Path to the merged model output directory.")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Max sequence length for loading.")
    parser.add_argument(
        "--maximum-memory-usage",
        type=float,
        default=0.6,
        help="Fraction of system memory Unsloth may use while saving merged weights.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.checkpoint,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
    )

    model.save_pretrained_merged(
        save_directory=args.output,
        tokenizer=tokenizer,
        save_method="merged_16bit",
        maximum_memory_usage=args.maximum_memory_usage,
    )


if __name__ == "__main__":
    main()
