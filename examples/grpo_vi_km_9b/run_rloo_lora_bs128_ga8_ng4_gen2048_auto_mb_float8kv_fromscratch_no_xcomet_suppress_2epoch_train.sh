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
export TRL_EXPERIMENTAL_SILENCE=1

export WANDB_PROJECT=${WANDB_PROJECT:-verl_grpo_example_mt}
export WANDB_NAME=${WANDB_NAME:-unsloth_vi_km_9b_rloo_lora_bs128_ga8_ng4_gen2048_auto_mb_float8kv_fromscratch_no_xcomet_suppress}
export WANDB_DIR=${WANDB_DIR:-/home/work/audio_sg16_data1/MT/haibo/GRPO/output/wandb}
mkdir -p "${WANDB_DIR}"
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY_FROM_SCRIPT="$(
    /home/work/audio_sg16_data1/MT/conda/envs/unsloth_train/bin/python - <<'PY'
from pathlib import Path
import re
for path in Path("/home/work/audio_sg16_data1/MT/haibo/GRPO/unsloth/examples/grpo_vi_km_9b").glob("run_lora_bs128_ga8*_train.sh"):
    text = path.read_text(errors="ignore")
    match = re.search(r'export WANDB_API_KEY="([^"]+)"', text)
    if match:
        print(match.group(1))
        break
PY
  )"
  if [[ -n "${WANDB_API_KEY_FROM_SCRIPT}" ]]; then
    export WANDB_API_KEY="${WANDB_API_KEY_FROM_SCRIPT}"
  fi
fi

export RM_URL=http://10.67.48.196:8910/score
unset XCOMET_URL
unset XCOMET_URLS
unset UNSLOTH_MT_XCOMET_URL
unset UNSLOTH_MT_XCOMET_URLS

# XCOMET is disabled for this speed run. Keep local BLEU/spBLEU/length/must_gt rewards.
unset UNSLOTH_MT_FAST_LOCAL_REWARD
export UNSLOTH_MT_DIRECT_XCOMET_BATCH=0
export UNSLOTH_MT_REWARD_WORKERS=128
export UNSLOTH_MT_XCOMET_BATCH_SIZE=512
export UNSLOTH_MT_XCOMET_BATCH_WAIT_MS=20
export UNSLOTH_MT_XCOMET_BATCH_QUEUE_SIZE=4096
export UNSLOTH_MT_XCOMET_TIMEOUT=120
export UNSLOTH_MT_XCOMET_MAX_RETRIES=1
export UNSLOTH_MT_XCOMET_RETRY_SLEEP=1

# RLOO-specific defaults. beta=0 avoids extra reference KL logprob work and matches
# the current speed-first GRPO setup more closely.
export UNSLOTH_RLOO_BETA=${UNSLOTH_RLOO_BETA:-0.0}
export UNSLOTH_RLOO_NORMALIZE_ADVANTAGES=${UNSLOTH_RLOO_NORMALIZE_ADVANTAGES:-0}

ROOT=/home/work/audio_sg16_data1/MT/haibo/GRPO/output/vi-km-9b-1.0/unsloth_rloo_lora_bs128_ga8_ng4_gen2048_auto_mb_float8kv_fromscratch_no_xcomet_suppress_2epoch
mkdir -p "${ROOT}/logs"

log_file="${ROOT}/logs/train_$(date +%Y%m%d_%H%M%S).log"
echo "[rloo-train] output_dir=${ROOT}"
echo "[rloo-train] log=${log_file}"
echo "[rloo-train] config: trainer=RLOO lora_rank=64 batch=128 grad_accum=8 effective_batch=1024 num_generations=4 generation_batch_size=2048 vllm_mem=0.40 float8_kv_cache=1 epochs=2 save_steps=80 save_total_limit=10 report_to=wandb wandb_project=${WANDB_PROJECT} wandb_name=${WANDB_NAME} xcomet=off sacrebleu_warning_suppressed=1 fast_local=0 reward_sample_log_count=2 wandb_reward_log=1 resume=none unsloth_grpo_mini_batch=auto rloo_beta=${UNSLOTH_RLOO_BETA}"

/home/work/audio_sg16_data1/MT/conda/envs/unsloth_train/bin/python -u train_rloo_vi_km_9b.py \
  --fast-inference \
  --float8-kv-cache \
  --lora-rank 64 \
  --model-name /home/work/audio_sg16_data1/MT/haibo/GRPO/model/vi-km-9b-1.0 \
  --train-parquet /home/work/audio_sg16_data1/MT/haibo/GRPO/train_data/grpo1.0/train.parquet \
  --output-dir "${ROOT}" \
  --reward-version v2 \
  --num-generations 4 \
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
  --comet-url http://10.67.48.196:8910/score \
  --xcomet-url "" \
  --comet-batch-size 1024 \
  --xcomet-batch-size 512 \
  --reward-sample-log-count 2 \
  --wandb-reward-log \
  --wandb-sample-count 2 \
  --report-to wandb 2>&1 | tee "${log_file}"
