#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/home/work/audio_sg16_data1/MT/haibo/GRPO/unsloth:${PYTHONPATH:-}
export UNSLOTH_DISABLE_STATISTICS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export TORCH_COMPILE_DISABLE=1
export RM_URL=http://10.67.48.196:8910/score
export XCOMET_URL=http://10.67.48.195:8910/score
export XCOMET_URLS=http://10.67.48.195:8910/score,http://10.67.48.152:8910/score
export UNSLOTH_MT_XCOMET_URLS="${XCOMET_URLS}"
export UNSLOTH_MT_REWARD_WORKERS=128

ROOT=/home/work/audio_sg16_data1/MT/haibo/GRPO/output/vi-km-9b-1.0/unsloth_grpo_full_vllm_batch_sweep
mkdir -p "${ROOT}/logs"

for train_batch in 256 128 64; do
  out_dir="${ROOT}/bs${train_batch}_gen1024"
  log_file="${ROOT}/logs/bs${train_batch}_$(date +%Y%m%d_%H%M%S).log"
  echo "[batch-sweep] trying per_device_train_batch_size=${train_batch}, generation_batch_size=1024"
  echo "[batch-sweep] output_dir=${out_dir}"
  echo "[batch-sweep] log=${log_file}"
  set +e
  /home/work/audio_sg16_data1/MT/conda/envs/unsloth_train/bin/python -u train_grpo_vi_km_9b.py \
    --full-finetune \
    --fast-inference \
    --model-name /home/work/audio_sg16_data1/MT/haibo/GRPO/model/vi-km-9b-1.0 \
    --train-parquet /home/work/audio_sg16_data1/MT/haibo/GRPO/train_data/grpo1.0/train.parquet \
    --output-dir "${out_dir}" \
    --reward-version v2 \
    --num-generations 8 \
    --generation-batch-size 1024 \
    --per-device-train-batch-size "${train_batch}" \
    --gradient-accumulation-steps 1 \
    --max-seq-length 1024 \
    --max-prompt-length 512 \
    --max-completion-length 256 \
    --gpu-memory-utilization 0.50 \
    --vllm-compilation-config 0 \
    --learning-rate 1e-6 \
    --max-steps 2 \
    --save-steps 400 \
    --logging-steps 1 \
    --group-normalize \
    --comet-url http://10.67.48.196:8910/score \
    --xcomet-url http://10.67.48.195:8910/score \
    --comet-batch-size 1024 \
    --xcomet-batch-size 512 \
    --report-to none 2>&1 | tee "${log_file}"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" -eq 0 ]]; then
    echo "[batch-sweep] SUCCESS train_batch=${train_batch} generation_batch_size=1024"
    exit 0
  fi
  echo "[batch-sweep] FAILED train_batch=${train_batch} rc=${rc}; trying next lower batch"
done

echo "[batch-sweep] all attempts failed"
exit 1
