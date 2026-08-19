import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Merge a PEFT LoRA checkpoint into a standalone HF model.")
    parser.add_argument("--checkpoint", required=True, help="Path to the LoRA checkpoint directory.")
    parser.add_argument("--output", required=True, help="Path to the merged model output directory.")
    parser.add_argument("--base-model", default=None, help="Base model path. Defaults to adapter_config.json.")
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.base_model:
        base_model = args.base_model
    else:
        adapter_config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
        base_model = adapter_config["base_model_name_or_path"]

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=False)
    merged = model.merge_and_unload()
    merged.save_pretrained(output, safe_serialization=True, max_shard_size=args.max_shard_size)

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True)
    tokenizer.save_pretrained(output)


if __name__ == "__main__":
    main()
