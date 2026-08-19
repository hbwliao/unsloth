"""RLOOTrainer subclass with the same verl-style GT injection used by GRPO.

The injection is done at token-id level after rollout generation and before
RLOO computes completion metrics, old logprobs, rewards, leave-one-out
advantages, and loss.
"""

from __future__ import annotations

from typing import Any, Optional

from trl import RLOOTrainer

from gt_inject_trainer import (
    _env_bool,
    _ground_truth_from_input,
    _is_must_gt_extra_info,
    _short_log_text,
)


class GTInjectRLOOTrainer(RLOOTrainer):
    """Unsloth/TRL RLOO trainer with verl-style GT injection."""

    def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        self._gt_inject_inputs = inputs
        try:
            return super()._generate_and_score_completions(inputs)
        finally:
            self._gt_inject_inputs = None

    def _generate_single_turn(self, prompts: list[str], images: Optional[list]):
        prompt_ids, completion_ids, forward_kwargs = super()._generate_single_turn(prompts, images)
        inputs = getattr(self, "_gt_inject_inputs", None)
        if inputs is None:
            return prompt_ids, completion_ids, forward_kwargs
        completion_ids = self._inject_ground_truth_completion_ids(inputs, completion_ids)
        return prompt_ids, completion_ids, forward_kwargs

    def _inject_ground_truth_completion_ids(
        self,
        inputs: list[dict[str, Any]],
        completion_ids: list[list[int]],
    ) -> list[list[int]]:
        n = int(getattr(self, "num_generations", 1) or 1)
        if n <= 1:
            return completion_ids
        bsz = min(len(completion_ids), len(inputs))
        num_groups = bsz // n
        if num_groups <= 0:
            return completion_ids

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

            injected_count += 1
            duplicate_injected_count += 1 if all_same else 0
            must_gt_injected_count += 1 if must_gt_input else 0

            if must_gt_input and _env_bool("REWARD_VERBOSE_LOG", False):
                src_text = extra_info.get("src_text", "") if isinstance(extra_info, dict) else ""
                print(
                    "\n"
                    + "=" * 96
                    + "\n"
                    + "[GT-Inject][Unsloth-RLOO] must_gt_input_data triggered GT injection.\n"
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
                "[GT-Inject][Unsloth-RLOO] Replaced last completion with ground_truth in "
                f"{injected_count}/{num_groups} groups "
                f"(duplicate={duplicate_injected_count}, must_gt={must_gt_injected_count})"
            )
        if must_gt_injected_count > 0:
            print(
                "[GT-Inject][Unsloth-RLOO] must_gt_input_data forced ground_truth replacement in "
                f"{must_gt_injected_count}/{num_groups} groups"
            )
        return completion_ids
