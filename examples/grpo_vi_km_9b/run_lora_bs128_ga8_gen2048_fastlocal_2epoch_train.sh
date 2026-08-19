#!/usr/bin/env bash
set -euo pipefail

cd /home/work/audio_sg16_data1/MT/haibo/GRPO/unsloth/examples/grpo_vi_km_9b

export PYTHONPATH=/home/work/audio_sg16_data1/MT/haibo/GRPO/unsloth:${PYTHONPATH:-}
export UNSLOTH_DISABLE_STATISTICS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export TORCH_COMPILE_DISABLE=1

export WANDB_PROJECT=${WANDB_PROJECT:-verl_grpo_example_mt}
export WANDB_NAME=${WANDB_NAME:-unsloth_vi_km_9b_lora_bs128_ga8_gen2048_fastlocal}
export WANDB_DIR=${WANDB_DIR:-/home/work/audio_sg16_data1/MT/haibo/GRPO/output/wandb}
mkdir -p "${WANDB_DIR}"
export WANDB_API_KEY="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"

unset RM_URL
unset UNSLOTH_MT_COMET_URL
unset XCOMET_URL
unset XCOMET_URLS
unset UNSLOTH_MT_XCOMET_URL
unset UNSLOTH_MT_XCOMET_URLS

# Fast local reward path: BLEU/spBLEU + length + must_gt only.
# This bypasses the full verl per-sample wrapper and suppresses sacrebleu warning noise.
export UNSLOTH_MT_FAST_LOCAL_REWARD=1
export UNSLOTH_MT_DIRECT_XCOMET_BATCH=0
export UNSLOTH_MT_REWARD_WORKERS=64
export UNSLOTH_MT_XCOMET_BATCH_SIZE=512
export UNSLOTH_MT_XCOMET_BATCH_WAIT_MS=20
export UNSLOTH_MT_XCOMET_BATCH_QUEUE_SIZE=4096
export UNSLOTH_MT_XCOMET_TIMEOUT=120
export UNSLOTH_MT_XCOMET_MAX_RETRIES=1
export UNSLOTH_MT_XCOMET_RETRY_SLEEP=1

ROOT=/home/work/audio_sg16_data1/MT/haibo/GRPO/output/vi-km-9b-1.0/unsloth_grpo_lora_bs128_ga8_gen2048_fastlocal_2epoch
mkdir -p "${ROOT}/logs"

log_file="${ROOT}/logs/train_$(date +%Y%m%d_%H%M%S).log"
echo "[lora-train] output_dir=${ROOT}"
echo "[lora-train] log=${log_file}"
echo "[lora-train] config: lora_rank=64 batch=128 grad_accum=8 effective_batch=1024 num_generations=8 generation_batch_size=2048 vllm_mem=0.40 epochs=2 save_steps=80 save_total_limit=10 report_to=wandb wandb_project=${WANDB_PROJECT} wandb_name=${WANDB_NAME} reward=fast_local_bleu_spbleu_length_mustgt comet=off xcomet=off"

/home/work/audio_sg16_data1/MT/conda/envs/unsloth_train/bin/python -u train_grpo_vi_km_9b.py \
  --fast-inference \
  --lora-rank 64 \
  --model-name /home/work/audio_sg16_data1/MT/haibo/GRPO/model/vi-km-9b-1.0 \
  --train-parquet /home/work/audio_sg16_data1/MT/haibo/GRPO/train_data/grpo1.0/train.parquet \
  --output-dir "${ROOT}" \
  --reward-version v2 \
  --num-generations 8 \
  --generation-batch-size 2048 \
  --per-device-train-batch-size 128 \
  --gradient-accumulation-steps 8 \
  --max-seq-length 1024 \
  --max-prompt-length 512 \
  --max-completion-length 256 \
  --gpu-memory-utilization 0.40 \
  --vllm-compilation-config 0 \
  --learning-rate 1e-6 \
  --max-steps -1 \
  --num-train-epochs 2 \
  --save-steps 80 \
  --save-total-limit 10 \
  --logging-steps 1 \
  --group-normalize \
  --comet-url "" \
  --xcomet-url "" \
  --comet-batch-size 1024 \
  --xcomet-batch-size 512 \
  --report-to wandb 2>&1 | tee "${log_file}"
