"""GRPOTrainer subclass that mirrors verl's ground-truth injection strategy.

The injection is done at token-id level after rollout generation and before
TRL computes logprobs, rewards, advantages, and loss.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from trl import GRPOTrainer


_MUST_GT_INPUT_CACHE: dict[tuple[str, int | None, int | None], set[str]] = {}


def _normalize_must_gt_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _short_log_text(value: Any, max_chars: int = 160) -> str:
    text = _normalize_must_gt_text(value)
    return text if len(text) <= max_chars else text[:max_chars] + "...(truncated)"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_must_gt_inputs(path: str | None = None) -> set[str]:
    path = str(
        path
        or os.environ.get("UNSLOTH_MT_MUST_GT_INPUT_PATH")
        or os.environ.get("MUST_GT_INPUT_PATH")
        or ""
    )
    if not path:
        return set()
    try:
        stat = os.stat(path)
        cache_key = (path, stat.st_mtime_ns, stat.st_size)
    except OSError:
        cache_key = (path, None, None)
    cached = _MUST_GT_INPUT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    values: set[str] = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                normalized = _normalize_must_gt_text(line)
                if normalized:
                    values.add(normalized)
    except OSError:
        values = set()
    _MUST_GT_INPUT_CACHE.clear()
    _MUST_GT_INPUT_CACHE[cache_key] = values
    return values


def _is_must_gt_extra_info(extra_info: Any) -> bool:
    if not isinstance(extra_info, dict):
        return False
    src_text = _normalize_must_gt_text(extra_info.get("src_text", ""))
    if not src_text:
        return False
    return src_text in _load_must_gt_inputs()


def _ground_truth_from_input(example: dict[str, Any]) -> str:
    reward_model = example.get("reward_model")
    if isinstance(reward_model, dict):
        for key in ("ground_truth", "reference", "answer", "chosen"):
            value = reward_model.get(key)
            if value not in (None, ""):
                return str(value)
    extra_info = example.get("extra_info")
    if isinstance(extra_info, dict):
        for key in ("ground_truth", "reference", "answer", "chosen"):
            value = extra_info.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


class GTInjectGRPOTrainer(GRPOTrainer):
    """Unsloth/TRL GRPO trainer with verl-style GT injection."""

    def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        self._gt_inject_inputs = inputs
        try:
            return super()._generate_and_score_completions(inputs)
        finally:
            self._gt_inject_inputs = None

    def _generate(self, prompts: list[str], images: Optional[list]):
        prompt_ids, completion_ids, total_completion_tokens, logprobs, forward_kwargs = super()._generate(prompts, images)
        inputs = getattr(self, "_gt_inject_inputs", None)
        if inputs is None:
            return prompt_ids, completion_ids, total_completion_tokens, logprobs, forward_kwargs
        completion_ids, logprobs = self._inject_ground_truth_completion_ids(inputs, completion_ids, logprobs)
        # Keep TRL's already-logged token count unchanged. This mirrors verl's
        # strategy closely enough for training; exact token-count metrics may
        # differ by the injected GT length.
        return prompt_ids, completion_ids, total_completion_tokens, logprobs, forward_kwargs

    def _inject_ground_truth_completion_ids(
        self,
        inputs: list[dict[str, Any]],
        completion_ids: list[list[int]],
        logprobs: list[list[float]] | None,
    ) -> tuple[list[list[int]], list[list[float]] | None]:
        n = int(getattr(self, "num_generations", 1) or 1)
        if n <= 1:
            return completion_ids, logprobs
        bsz = min(len(completion_ids), len(inputs))
        num_groups = bsz // n
        if num_groups <= 0:
            return completion_ids, logprobs

        eos_token_id = self.eos_token_id
        if eos_token_id is None:
            eos_token_id = getattr(self.processing_class, "eos_token_id", None)
        max_response_length = int(getattr(self, "max_completion_length", 0) or 0)

        injected_count = 0
        duplicate_injected_count = 0
        must_gt_injected_count = 0

        for group_idx in range(num_groups):
            start_idx = group_idx * n
            end_idx = start_idx + n
            group = completion_ids[start_idx:end_idx]
            if len(group) != n:
                continue

            all_same = all(group[0] == group[i] for i in range(1, n))
            extra_info = inputs[start_idx].get("extra_info")
            must_gt_input = _is_must_gt_extra_info(extra_info)
            if not all_same and not must_gt_input:
                continue

            gt_text = _ground_truth_from_input(inputs[start_idx])
            if not gt_text.strip():
                continue

            gt_tokens = self.processing_class.encode(gt_text, add_special_tokens=False)
            if eos_token_id is not None:
                gt_tokens = list(gt_tokens) + [int(eos_token_id)]
            else:
                gt_tokens = list(gt_tokens)
            if max_response_length > 0:
                gt_tokens = gt_tokens[:max_response_length]
            if not gt_tokens:
                continue

            replace_idx = end_idx - 1
            completion_ids[replace_idx] = gt_tokens
            if logprobs is not None and replace_idx < len(logprobs):
                # The injected sequence was not sampled by vLLM. The training
                # script disables vLLM importance sampling correction by default;
                # these zeros are only a shape-safe fallback if someone enables it.
                logprobs[replace_idx] = [0.0] * len(gt_tokens)

            injected_count += 1
            duplicate_injected_count += 1 if all_same else 0
            must_gt_injected_count += 1 if must_gt_input else 0

            if must_gt_input and _env_bool("REWARD_VERBOSE_LOG", False):
                src_text = extra_info.get("src_text", "") if isinstance(extra_info, dict) else ""
                print(
                    "\n"
                    + "=" * 96
                    + "\n"
                    + "[GT-Inject][Unsloth] must_gt_input_data triggered GT injection.\n"
                    + f"global_step={getattr(self.state, 'global_step', 'unknown')}\n"
                    + f"group_idx={group_idx}, replace_idx={replace_idx}, rollout_n={n}\n"
                    + f"duplicate_group={all_same}\n"
                    + f"ground_truth_chars={len(gt_text)}, ground_truth_tokens={len(gt_tokens)}\n"
                    + f"src_text={_short_log_text(src_text)}\n"
                    + "=" * 96
                    + "\n"
                )

        if injected_count > 0:
            print(
                "[GT-Inject][Unsloth] Replaced last completion with ground_truth in "
                f"{injected_count}/{num_groups} groups "
                f"(duplicate={duplicate_injected_count}, must_gt={must_gt_injected_count})"
            )
        if must_gt_injected_count > 0:
            print(
                "[GT-Inject][Unsloth] must_gt_input_data forced ground_truth replacement in "
                f"{must_gt_injected_count}/{num_groups} groups"
            )
        return completion_ids, logprobs

