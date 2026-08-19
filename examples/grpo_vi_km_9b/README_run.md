# Unsloth GRPO VI->KM reward migration

## 路径

```bash
cd /home/work/audio_sg16_data1/MT/haibo/GRPO/unsloth/examples/grpo_vi_km_9b
```

## 激活环境

```bash
source /home/work/audio_sg16_data1/MT/conda/bin/activate
conda activate unsloth_train
```

## Reward 版本

### v1：静态融合

- BLEU
- spBLEU
- length
- COMET HTTP
- XCOMET HTTP
- 关闭 language detection
- 不做 DVAO 动态聚合

### v2：must_gt + 日志 + 可选组内归一化

在 v1 基础上增加：

- must_gt input 匹配
- must_gt reference anchor 调整
- reward component JSONL 日志
- 可选 reward-level group normalization

注意：TRL GRPOTrainer 自身已经会按 `num_generations` 做 group advantage normalization。通常不建议再打开 `--group-normalize`，除非你明确想让 reward 数值本身也先做一次组内标准化。

### v3：近似 DVAO

在 v2 基础上增加：

- 在 reward function 内按 rollout group 计算组件方差
- 根据组件方差动态分配 BLEU/spBLEU/length/COMET/XCOMET 权重
- 近似复刻 verl `RayPPOTrainer._adjust_dvao_reward_scores`

限制：Unsloth/TRL reward function 拿不到 verl 的完整 `DataProto` 和 reward tensor 位置，因此这里返回的是每个 completion 的最终 scalar reward，不再修改 token-level reward tensor。

## 默认训练路径

`train_grpo_vi_km_9b.py` 默认启用 Unsloth `fast_inference=True`，也就是优先走 vLLM 高速 rollout 路径。只有显式加：

```bash
--no-fast-inference
```

才会退回普通 generation。这个兜底只用于排查问题，不是推荐训练路径。

当前已验证可 import 的高速组合：

```text
torch        2.9.0+cu128
vllm         0.11.1
trl          0.24.0
transformers 4.57.6
GPU          NVIDIA B300 SXM6 AC MIG 7g.269gb
trainer      UnslothGRPOTrainer
```

## 与当前 verl 配置的对齐

当前脚本默认按“尽量贴近 verl”配置：

```text
reward-version                 v3
num-generations                8
target-rollout-groups          1024
generation-batch-size          8192
max-prompt-length              512
max-completion-length          512
learning-rate                  1e-6
save-steps                     400
gpu-memory-utilization         0.30
fast-inference                 True
gradient-checkpointing         False
language reward                disabled
```

对应当前 verl 日志：

```text
ROLLOUT_N                      8
RewardStepSummary samples      8192
RewardStepSummary groups       1024
MAX_PROMPT_LENGTH              512
MAX_RESPONSE_LENGTH            512
ACTOR_LR                       1e-6
SAVE_FREQ                      400
ROLLOUT_GPU_MEMORY_UTILIZATION 0.30
LANGUAGE_REWARD_ENABLED        False
```

`gradient-checkpointing` 默认关闭，因为当前 B300 显存足够，关闭它通常能减少反传重算、提升 `update_actor` 速度。只有显存不够时再显式加：

```bash
--gradient-checkpointing
```

GT-Inject 已经做成 trainer 级实现：

```text
GTInjectGRPOTrainer
```

它会在生成完成后、计算 logprob/reward/loss 前，把每个 rollout group 的最后一个 completion token ids 替换为 ground_truth token ids。注入条件对齐 verl：

```text
1. 组内所有 response 完全相同
2. 或该 prompt 命中 must_gt_input
```

仍然不完全一致的地方：

- verl 的 GT-Inject 会同步更新 verl DataProto 里的 input_ids / attention_mask / position_ids / rollout_log_probs；Unsloth 版是在 TRL 的 list-of-token-ids 阶段替换 completion_ids，后续 attention_mask/logprob/reward/loss 由 TRL 重新构造。
- verl 的 DVAO 修改 token-level reward tensor；当前 Unsloth v3 返回 scalar reward，TRL 再做 group advantage。
- verl 是 Ray 多节点；Unsloth 脚本是本地 Unsloth/TRL 训练，不是 Ray trainer。
- 对注入 GT 的样本，vLLM 并没有真实采样 logprob；因此脚本默认关闭 `vllm_importance_sampling_correction`，避免对 GT 注入样本做错误 off-policy correction。

## 试跑 v1

```bash
python train_grpo_vi_km_9b.py \
  --reward-version v1 \
  --model-name /home/work/audio_sg16_data1/MT/haibo/GRPO/model/vi-km-9b-1.0 \
  --train-parquet /home/work/audio_sg16_data1/MT/haibo/GRPO/train_data/grpo1.0/train.parquet \
  --output-dir /home/work/audio_sg16_data1/MT/haibo/GRPO/output/vi-km-9b-1.0/unsloth_grpo_v1 \
  --num-generations 8 \
  --per-device-train-batch-size 8 \
  --max-steps 20 \
  --comet-url http://10.67.48.196:8910/score \
  --xcomet-url http://10.67.48.195:8910/score
```

注意：如果不想按 verl 的 1024 groups 大 generation batch 试跑，可以临时加小参数：

```bash
--target-rollout-groups 1 --max-steps 1
```

## 试跑 v2

```bash
python train_grpo_vi_km_9b.py \
  --reward-version v2 \
  --output-dir /home/work/audio_sg16_data1/MT/haibo/GRPO/output/vi-km-9b-1.0/unsloth_grpo_v2 \
  --max-steps 20 \
  --comet-url http://10.67.48.196:8910/score \
  --xcomet-url http://10.67.48.195:8910/score
```

## 试跑 v3

```bash
python train_grpo_vi_km_9b.py \
  --reward-version v3 \
  --output-dir /home/work/audio_sg16_data1/MT/haibo/GRPO/output/vi-km-9b-1.0/unsloth_grpo_v3 \
  --max-steps 20 \
  --comet-url http://10.67.48.196:8910/score \
  --xcomet-url http://10.67.48.195:8910/score
```

## Reward 权重

可以通过环境变量覆盖默认权重：

```bash
export UNSLOTH_MT_BLEU_WEIGHT=0.20
export UNSLOTH_MT_SPBLEU_WEIGHT=0.20
export UNSLOTH_MT_LENGTH_WEIGHT=0.20
export UNSLOTH_MT_COMET_WEIGHT=0.20
export UNSLOTH_MT_XCOMET_WEIGHT=0.20
```

## 日志

默认写到：

```bash
<output-dir>/reward_components.jsonl
```

也可以指定：

```bash
export UNSLOTH_MT_REWARD_LOG_PATH=/tmp/unsloth_mt_reward_components.jsonl
```

## 当前环境注意

之前 `unsloth_train` 里出现过 `trl 0.24.0` 与 `vllm 0.23.0+cu129` 的接口不兼容：

```text
cannot import name 'GuidedDecodingParams' from 'vllm.sampling_params'
```

已经调整为 `vllm 0.11.1 + torch 2.9.0+cu128`，满足 B300/SM100 上 Unsloth 的 torch>=2.9 要求，并保留 TRL 需要的 `GuidedDecodingParams` 接口。

如果后续升级 vLLM/TRL，必须重新验证：

```bash
python - <<'PY'
import unsloth, torch, vllm, trl
from vllm.sampling_params import GuidedDecodingParams
from trl import GRPOTrainer
print(torch.__version__, vllm.__version__, trl.__version__, GRPOTrainer.__name__)
PY
```
