#!/usr/bin/env python
"""Unsloth RLOO entrypoint for the VI->KM MT reward migration.

This mirrors train_grpo_vi_km_9b.py as closely as possible while replacing
GRPO's group advantage with RLOO's leave-one-out baseline.
"""

from __future__ import annotations

import os
from pathlib import Path

from train_grpo_vi_km_9b import StepTimingCallback, parse_args


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    args = parse_args()
    if args.full_finetune and args.load_in_4bit:
        raise ValueError("--full-finetune cannot be combined with --load-in-4bit.")

    # Reward adapter configuration. Keep language reward disabled by design.
    os.environ["UNSLOTH_MT_NUM_GENERATIONS"] = str(args.num_generations)
    os.environ["UNSLOTH_MT_GROUP_NORMALIZE"] = "1" if args.group_normalize else "0"
    os.environ["UNSLOTH_MT_COMET_URL"] = args.comet_url
    os.environ["UNSLOTH_MT_XCOMET_URL"] = args.xcomet_url
    os.environ["UNSLOTH_MT_COMET_BATCH_SIZE"] = str(args.comet_batch_size)
    os.environ["UNSLOTH_MT_COMET_BATCH_WAIT_MS"] = str(args.comet_batch_wait_ms)
    os.environ["UNSLOTH_MT_COMET_BATCH_QUEUE_SIZE"] = str(args.comet_batch_queue_size)
    os.environ["UNSLOTH_MT_XCOMET_BATCH_SIZE"] = str(args.xcomet_batch_size)
    os.environ["UNSLOTH_MT_XCOMET_BATCH_WAIT_MS"] = str(args.xcomet_batch_wait_ms)
    os.environ["UNSLOTH_MT_XCOMET_BATCH_QUEUE_SIZE"] = str(args.xcomet_batch_queue_size)
    os.environ["UNSLOTH_MT_MUST_GT_INPUT_PATH"] = args.must_gt_input_path
    os.environ["UNSLOTH_MT_SAMPLE_LOG_COUNT"] = str(max(0, args.reward_sample_log_count))
    os.environ["UNSLOTH_MT_WANDB_REWARD_LOG"] = (
        "1" if args.wandb_reward_log and "wandb" in str(args.report_to).lower() else "0"
    )
    os.environ["UNSLOTH_MT_WANDB_SAMPLE_COUNT"] = str(max(0, args.wandb_sample_count))
    os.environ["UNSLOTH_ENABLE_FULL_FINETUNING"] = "1" if args.full_finetune else "0"
    os.environ.setdefault(
        "UNSLOTH_MT_REWARD_LOG_PATH",
        str(Path(args.output_dir) / "reward_components.jsonl"),
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

    rloo_beta = float(os.environ.get("UNSLOTH_RLOO_BETA", "0.0"))
    rloo_normalize_advantages = _env_bool("UNSLOTH_RLOO_NORMALIZE_ADVANTAGES", False)

    # Unsloth must be imported before TRL so trainer patches are applied.
    import unsloth  # noqa: F401

    # The current local env may contain a vLLM version newer than TRL supports.
    # If we are not using Unsloth fast_inference/vLLM, hide vLLM from TRL so
    # RLOOTrainer can fall back to regular generation.
    if not args.fast_inference:
        import trl.import_utils as trl_import_utils

        trl_import_utils._vllm_available = False

    from datasets import load_dataset
    from trl import RLOOConfig
    from unsloth import FastLanguageModel

    from gt_inject_rloo_trainer import GTInjectRLOOTrainer
    from reward_mt_adapter import get_reward_func

    load_kwargs = dict(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=args.full_finetune,
        fast_inference=args.fast_inference,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )
    if args.fast_inference:
        load_kwargs.update(
            gpu_memory_utilization=args.gpu_memory_utilization,
            float8_kv_cache=args.float8_kv_cache,
            enforce_eager=True,
            compilation_config=args.vllm_compilation_config,
        )
    if not args.full_finetune:
        load_kwargs["max_lora_rank"] = args.lora_rank

    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

    if args.full_finetune:
        model = FastLanguageModel.for_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=args.lora_rank,
            use_gradient_checkpointing=args.gradient_checkpointing,
            random_state=3407,
        )

    # Gemma2 configs can carry cache_implementation="hybrid". Unsloth disables
    # use_cache for training, and Transformers refuses to save that combination.
    # Clear it so checkpoint/final saves do not fail after a successful step.
    for config_obj in (getattr(model, "generation_config", None), getattr(model, "config", None)):
        if config_obj is not None and getattr(config_obj, "use_cache", None) is False:
            if getattr(config_obj, "cache_implementation", None) is not None:
                config_obj.cache_implementation = None

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[config] trainer=RLOO full_finetune={args.full_finetune} "
        f"fast_inference={args.fast_inference} "
        f"gradient_checkpointing={args.gradient_checkpointing} "
        f"rloo_beta={rloo_beta} "
        f"rloo_normalize_advantages={rloo_normalize_advantages} "
        f"trainable_params={trainable_params:,}/{total_params:,} "
        f"({trainable_params / max(total_params, 1):.2%})"
    )

    dataset = load_dataset("parquet", data_files=args.train_parquet, split="train")
    reward_func = get_reward_func(args.reward_version)

    generation_batch_size = args.generation_batch_size
    if generation_batch_size is None and args.steps_per_generation is None:
        generation_batch_size = args.target_rollout_groups * args.num_generations

    training_args = RLOOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        generation_batch_size=generation_batch_size,
        steps_per_generation=args.steps_per_generation,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        max_grad_norm=0.1,
        report_to=args.report_to,
        bf16=True,
        seed=3407,
        beta=rloo_beta,
        normalize_advantages=rloo_normalize_advantages,
        vllm_gpu_memory_utilization=args.gpu_memory_utilization,
        unsloth_grpo_mini_batch=args.unsloth_grpo_mini_batch,
        unsloth_logit_chunk_multiplier=args.unsloth_logit_chunk_multiplier,
    )

    trainer = GTInjectRLOOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=dataset,
    )
    if not args.no_step_time_log:
        trainer.add_callback(StepTimingCallback())
    resume_from_checkpoint = str(args.resume_from_checkpoint or "").strip()
    if resume_from_checkpoint.lower() == "latest":
        checkpoints = sorted(
            Path(args.output_dir).glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[-1]) if path.name.rsplit("-", 1)[-1].isdigit() else -1,
        )
        resume_from_checkpoint = str(checkpoints[-1]) if checkpoints else ""
    if resume_from_checkpoint:
        print(f"[resume] resume_from_checkpoint={resume_from_checkpoint}", flush=True)
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    final_dir = Path(args.output_dir) / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
