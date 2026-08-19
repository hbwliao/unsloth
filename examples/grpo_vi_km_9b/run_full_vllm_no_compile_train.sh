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

/home/work/audio_sg16_data1/MT/conda/envs/unsloth_train/bin/python -u train_grpo_vi_km_9b.py \
  --full-finetune \
  --fast-inference \
  --model-name /home/work/audio_sg16_data1/MT/haibo/GRPO/model/vi-km-9b-1.0 \
  --train-parquet /home/work/audio_sg16_data1/MT/haibo/GRPO/train_data/grpo1.0/train.parquet \
  --output-dir /home/work/audio_sg16_data1/MT/haibo/GRPO/output/vi-km-9b-1.0/unsloth_grpo_full_vllm_train \
  --reward-version v2 \
  --num-generations 8 \
  --generation-batch-size 64 \
  --per-device-train-batch-size 8 \
  --gradient-accumulation-steps 1 \
  --max-seq-length 1024 \
  --max-prompt-length 512 \
  --max-completion-length 256 \
  --gpu-memory-utilization 0.30 \
  --vllm-compilation-config 0 \
  --learning-rate 1e-6 \
  --max-steps 100664 \
  --save-steps 400 \
  --logging-steps 1 \
  --group-normalize \
  --comet-url http://10.67.48.196:8910/score \
  --xcomet-url http://10.67.48.195:8910/score \
  --comet-batch-size 1024 \
  --xcomet-batch-size 1024 \
  --report-to none
