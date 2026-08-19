#!/usr/bin/env python
"""Unsloth GRPO entrypoint for the VI->KM MT reward migration.

This is intentionally a local Unsloth/TRL script, not a Ray submission script.
Unsloth GRPO does not provide verl-style Ray trainer orchestration.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        default="/home/work/audio_sg16_data1/MT/haibo/GRPO/model/vi-km-9b-1.0",
    )
    parser.add_argument(
        "--train-parquet",
        default="/home/work/audio_sg16_data1/MT/haibo/GRPO/train_data/grpo1.0/train.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/work/audio_sg16_data1/MT/haibo/GRPO/output/vi-km-9b-1.0/unsloth_grpo",
    )
    parser.add_argument("--reward-version", default="v3", choices=["v1", "v2", "v3"])
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument(
        "--target-rollout-groups",
        type=int,
        default=1024,
        help="Target prompt groups per generation batch. 1024 groups * 8 generations matches current verl logs.",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=None,
        help="TRL generation_batch_size. Defaults to target_rollout_groups * num_generations.",
    )
    parser.add_argument(
        "--steps-per-generation",
        type=int,
        default=None,
        help="Optional TRL steps_per_generation. Mutually exclusive with generation_batch_size.",
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=400)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument(
        "--full-finetune",
        action="store_true",
        help="Train all base model parameters. This disables LoRA/PEFT wrapping.",
    )
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument(
        "--float8-kv-cache",
        action="store_true",
        help="Enable vLLM FP8 KV cache through Unsloth FastLanguageModel.from_pretrained.",
    )
    parser.add_argument(
        "--vllm-compilation-config",
        type=int,
        default=3,
        help="Passed to Unsloth's vLLM loader. Use 0 to disable vLLM inductor compilation.",
    )
    parser.add_argument("--fast-inference", dest="fast_inference", action="store_true", default=True)
    parser.add_argument("--no-fast-inference", dest="fast_inference", action="store_false")
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--group-normalize", action="store_true")
    parser.add_argument("--comet-url", default=os.environ.get("RM_URL", ""))
    parser.add_argument("--xcomet-url", default=os.environ.get("XCOMET_URL", ""))
    parser.add_argument("--comet-batch-size", type=int, default=int(os.environ.get("UNSLOTH_MT_COMET_BATCH_SIZE", "1024")))
    parser.add_argument("--comet-batch-wait-ms", type=int, default=int(os.environ.get("UNSLOTH_MT_COMET_BATCH_WAIT_MS", "20")))
    parser.add_argument("--comet-batch-queue-size", type=int, default=int(os.environ.get("UNSLOTH_MT_COMET_BATCH_QUEUE_SIZE", "2048")))
    parser.add_argument("--xcomet-batch-size", type=int, default=int(os.environ.get("UNSLOTH_MT_XCOMET_BATCH_SIZE", "1024")))
    parser.add_argument("--xcomet-batch-wait-ms", type=int, default=int(os.environ.get("UNSLOTH_MT_XCOMET_BATCH_WAIT_MS", "20")))
    parser.add_argument("--xcomet-batch-queue-size", type=int, default=int(os.environ.get("UNSLOTH_MT_XCOMET_BATCH_QUEUE_SIZE", "2048")))
    parser.add_argument(
        "--must-gt-input-path",
        default="/home/work/audio_sg16_data1/MT/haibo/GRPO/verl/verl/examples/grpo_trainer/vi-km-9b/vi_km_must_gt.txt",
    )
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--no-step-time-log", action="store_true")
    parser.add_argument(
        "--reward-sample-log-count",
        type=int,
        default=2,
        help="Number of reward examples to print per rollout in the local log.",
    )
    parser.add_argument(
        "--wandb-reward-log",
        action="store_true",
        default=True,
        help="Log reward component curves to W&B when W&B is active.",
    )
    parser.add_argument(
        "--no-wandb-reward-log",
        dest="wandb_reward_log",
        action="store_false",
    )
    parser.add_argument(
        "--wandb-sample-count",
        type=int,
        default=2,
        help="Number of reward examples to log to W&B table per rollout.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default="",
        help='Checkpoint path to resume from, or "latest" to use the newest checkpoint under output_dir.',
    )
    parser.add_argument(
        "--unsloth-grpo-mini-batch",
        type=int,
        default=None,
        help="Mini batch size used by Unsloth when recomputing GRPO logprobs.",
    )
    parser.add_argument(
        "--unsloth-logit-chunk-multiplier",
        type=int,
        default=None,
        help="Chunk multiplier for Unsloth selective log-softmax/logit computation.",
    )
    return parser.parse_args()


class StepTimingCallback:
    """Print optimizer-step wall time into the normal training log."""

    def __init__(self) -> None:
        self._train_start: float | None = None
        self._step_start: float | None = None
        self._last_step = 0

    def __getattr__(self, name: str):
        if name.startswith("on_"):
            return lambda *args, **kwargs: None
        raise AttributeError(name)

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        self._train_start = time.monotonic()
        self._step_start = self._train_start

    def on_step_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        self._step_start = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        now = time.monotonic()
        step = int(getattr(state, "global_step", 0) or 0)
        if step <= self._last_step:
            return
        step_elapsed = now - (self._step_start or now)
        total_elapsed = now - (self._train_start or now)
        avg_elapsed = total_elapsed / max(step, 1)
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        eta_seconds = avg_elapsed * max(max_steps - step, 0) if max_steps > 0 else 0.0
        print(
            "[Unsloth-Step-Time] "
            f"step={step}/{max_steps or 'unknown'} "
            f"step_sec={step_elapsed:.2f} "
            f"avg_sec={avg_elapsed:.2f} "
            f"elapsed_h={total_elapsed / 3600:.3f} "
            f"eta_h={eta_seconds / 3600:.3f}",
            flush=True,
        )
        self._last_step = step


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
    # B300/MIG reports sm_103a. Triton/ptxas bundled in some Python wheels can
    # be CUDA 12.8 and reject that target, while the system CUDA 12.9 ptxas
    # supports it. Set this before importing Unsloth/vLLM/Triton.
    os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")

    # Unsloth must be imported before TRL so trainer patches are applied.
    import unsloth  # noqa: F401

    # The current local env may contain a vLLM version newer than TRL supports.
    # If we are not using Unsloth fast_inference/vLLM, hide vLLM from TRL so
    # GRPOTrainer can fall back to regular generation.
    if not args.fast_inference:
        import trl.import_utils as trl_import_utils

        trl_import_utils._vllm_available = False

    from datasets import load_dataset
    from trl import GRPOConfig
    from unsloth import FastLanguageModel

    from gt_inject_trainer import GTInjectGRPOTrainer
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
        f"[config] full_finetune={args.full_finetune} "
        f"fast_inference={args.fast_inference} "
        f"gradient_checkpointing={args.gradient_checkpointing} "
        f"trainable_params={trainable_params:,}/{total_params:,} "
        f"({trainable_params / max(total_params, 1):.2%})"
    )

    dataset = load_dataset("parquet", data_files=args.train_parquet, split="train")
    reward_func = get_reward_func(args.reward_version)

    generation_batch_size = args.generation_batch_size
    if generation_batch_size is None and args.steps_per_generation is None:
        generation_batch_size = args.target_rollout_groups * args.num_generations

    training_args = GRPOConfig(
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
        vllm_importance_sampling_correction=False,
        unsloth_grpo_mini_batch=args.unsloth_grpo_mini_batch,
        unsloth_logit_chunk_multiplier=args.unsloth_logit_chunk_multiplier,
    )

    trainer = GTInjectGRPOTrainer(
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
