"""Unsloth/TRL GRPO reward adapter for the VI->KM MT task.

This file intentionally reuses the existing verl reward implementation instead
of copying BLEU / COMET / XCOMET / length logic into the Unsloth tree.

Reward versions:
  - mt_reward_static_v1:
      BLEU/spBLEU + length + COMET + XCOMET, language reward disabled,
      static available-component weighted fusion.
  - mt_reward_mustgt_v2:
      v1 + must_gt anchor adjustment + component summary logging +
      optional reward-level group normalization.
  - mt_reward_dvao_v3:
      v2 + lightweight DVAO group-variance dynamic weighting inside TRL's
      reward function. This approximates verl's trainer-side DVAO because TRL
      only exposes reward-function-level batch context.

TRL calls reward functions as:
    reward_func(prompts=..., completions=..., completion_ids=..., **dataset_columns)

The expected dataset columns are the same as the current verl parquet:
    data_source, prompt, reward_model, extra_info
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except Exception:  # pragma: no cover - keep reward importable for static checks.
    np = None


DEFAULT_VERL_REWARD_DIR = Path(
    "/home/work/audio_sg16_data1/MT/haibo/GRPO/verl/verl/verl/utils/reward_score"
)
DEFAULT_LOG_PATH = Path(
    "/home/work/audio_sg16_data1/MT/haibo/GRPO/unsloth/examples/grpo_vi_km_9b/reward_logs/reward_components.jsonl"
)

_MODULE_CACHE: dict[str, Any] = {}
_FAST_SPBLEU_METRIC: Any = None


def _env(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name, "")
    if raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = _env(name, "")
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = _env(name, "")
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _split_urls(value: str) -> list[str]:
    return [x.strip() for x in re.split(r"[\s,;]+", value or "") if x.strip()]


def _xcomet_urls() -> list[str]:
    raw = _env("UNSLOTH_MT_XCOMET_URLS", _env("XCOMET_URLS", ""))
    urls = _split_urls(raw)
    if urls:
        return urls
    single = _env("UNSLOTH_MT_XCOMET_URL", _env("XCOMET_URL", ""))
    return _split_urls(single)


def _xcomet_url_for(sample_index: int = 0) -> str:
    urls = _xcomet_urls()
    if not urls:
        return ""
    return urls[int(sample_index) % len(urls)]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _clip01(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value, 0.0)))


def _load_module(path: Path, module_name: str) -> Any:
    path = path.resolve()
    cache_key = f"{module_name}:{path}"
    cached = _MODULE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not path.exists():
        raise FileNotFoundError(f"Reward module not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reward module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[cache_key] = module
    return module


def _suppress_sacrebleu_warning_noise() -> None:
    """Keep reward logs useful and avoid per-sample sacrebleu warning overhead."""

    warnings.filterwarnings(
        "ignore",
        message=r".*Tokenizer 'spm' has been changed to 'flores101'.*",
        category=Warning,
    )
    for logger_name in ("sacrebleu", "sacrebleu.metrics", "sacrebleu.tokenizers"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def _fast_spbleu_sentence_score(pred: str, answer: str) -> float | None:
    """Sentence-level spBLEU without sacrebleu's deprecated 'spm' warning path."""

    global _FAST_SPBLEU_METRIC
    try:
        if _FAST_SPBLEU_METRIC is None:
            from sacrebleu.metrics import BLEU

            try:
                _FAST_SPBLEU_METRIC = BLEU(tokenize="flores101", smooth_method="exp", effective_order=True)
            except TypeError:
                _FAST_SPBLEU_METRIC = BLEU(tokenize="flores101", smooth_method="exp")
        return float(_FAST_SPBLEU_METRIC.sentence_score(pred, [answer]).score) / 100.0
    except Exception:
        return None


def _install_reward_fast_patches(module: Any) -> None:
    _suppress_sacrebleu_warning_noise()
    base = getattr(module, "_BASE_MODULE", None)
    if base is None or getattr(base, "_UNSLOTH_FAST_PATCHED", False):
        return

    def _patched_sacrebleu_spbleu_reward(pred: str, answer: str) -> float | None:
        return _fast_spbleu_sentence_score(str(pred or ""), str(answer or ""))

    base._sacrebleu_spbleu_reward = _patched_sacrebleu_spbleu_reward
    base._UNSLOTH_FAST_PATCHED = True


def _reward_dir() -> Path:
    return Path(_env("UNSLOTH_MT_VERL_REWARD_DIR", str(DEFAULT_VERL_REWARD_DIR)))


def _dvao_module() -> Any:
    module = _load_module(_reward_dir() / "mt_bpo_br_id_commet3.0_dvao.py", "unsloth_verl_mt_dvao")
    _install_reward_fast_patches(module)
    return module


def _prewarm_verl_caches(module: Any) -> None:
    must_gt_input_path = _env("UNSLOTH_MT_MUST_GT_INPUT_PATH", _env("MUST_GT_INPUT_PATH", ""))
    if not must_gt_input_path:
        return
    for attr_name in ("_BASE_MODULE", "_XCOMET_MODULE"):
        child = getattr(module, attr_name, None)
        loader = getattr(child, "_load_must_gt_inputs", None)
        if callable(loader):
            loader(must_gt_input_path)


def _configure_verl_switches(module: Any, *, include_xcomet: bool = True) -> None:
    """Set switches on the imported module object without modifying source files."""

    comet_url = _env("UNSLOTH_MT_COMET_URL", _env("RM_URL", ""))
    xcomet_enabled = bool(_xcomet_urls()) and include_xcomet
    module.bleu_reward = 1
    module.spbleu_reward = 1
    module.length_reward = 1
    module.commet_reward = 1 if comet_url else 0
    module.xcomet_reward = 1 if xcomet_enabled else 0
    module.lange_detect_reward = 0  # explicitly disabled for all migrated versions
    module.language_reward = 0 if hasattr(module, "language_reward") else 0
    module.llm_reward = 0
    module.must_gt_reward = 1
    module.dvao_reward = 1


def _completion_to_text(completion: Any) -> str:
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", completion.get("text", "")))
    if isinstance(completion, (list, tuple)):
        # TRL conversational completions are often [{"role": "assistant", "content": "..."}].
        for item in reversed(completion):
            if isinstance(item, dict) and ("content" in item or "text" in item):
                return str(item.get("content", item.get("text", "")))
        return "".join(_completion_to_text(item) for item in completion)
    return str(completion)


def _stable_prompt_key(prompt: Any) -> str:
    try:
        return json.dumps(prompt, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return repr(prompt)


def _as_list(value: Any, n: int, default: Any = None) -> list[Any]:
    if isinstance(value, list):
        if len(value) == n:
            return value
        if len(value) == 1 and n > 1:
            return value * n
    if isinstance(value, tuple):
        value = list(value)
        if len(value) == n:
            return value
    return [default for _ in range(n)]


def _dict_or_empty(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _ground_truth_for(i: int, reward_models: list[Any], extra_infos: list[dict[str, Any]], explicit: list[Any]) -> str:
    if i < len(explicit) and explicit[i] not in (None, ""):
        return str(explicit[i])
    rm = _dict_or_empty(reward_models[i]) if i < len(reward_models) else {}
    for key in ("ground_truth", "reference", "answer", "chosen"):
        if rm.get(key) not in (None, ""):
            return str(rm[key])
    extra = extra_infos[i] if i < len(extra_infos) else {}
    for key in ("ground_truth", "reference", "answer", "chosen"):
        if extra.get(key) not in (None, ""):
            return str(extra[key])
    return ""


def _available_component(record: dict[str, Any], name: str, default_weight: float) -> tuple[float, bool, float]:
    score_key = f"dvao_{name}_score"
    avail_key = f"dvao_{name}_available"
    weight_key = f"dvao_static_{name}_weight"
    available = _safe_float(record.get(avail_key, 0.0), 0.0) > 0.5
    score = _clip01(record.get(score_key, 0.0))
    weight = max(0.0, _safe_float(record.get(weight_key, default_weight), default_weight))
    return score, available, weight


def _weighted_available_score(components: Iterable[tuple[float, bool, float]]) -> float:
    numerator = 0.0
    denominator = 0.0
    for score, available, weight in components:
        if available and weight > 0:
            numerator += score * weight
            denominator += weight
    if denominator <= 1e-8:
        return 0.0
    return numerator / denominator


def _component_weights() -> dict[str, float]:
    return {
        "bleu": _env_float("UNSLOTH_MT_BLEU_WEIGHT", 0.20),
        "spbleu": _env_float("UNSLOTH_MT_SPBLEU_WEIGHT", 0.20),
        "length": _env_float("UNSLOTH_MT_LENGTH_WEIGHT", 0.20),
        "commet": _env_float("UNSLOTH_MT_COMET_WEIGHT", 0.20),
        "xcomet": _env_float("UNSLOTH_MT_XCOMET_WEIGHT", 0.20),
    }


def _normalize_linear(value: Any, lo: float, hi: float) -> float:
    value = _safe_float(value, lo)
    if hi <= lo:
        return 0.0
    return _clip01((value - lo) / (hi - lo))


@lru_cache(maxsize=262_144)
def _cached_bleu_score(pred: str, answer: str, target_lang: str, tokenizer_mode: str) -> tuple[float | None, str]:
    module = _dvao_module()
    base = getattr(module, "_BASE_MODULE", None)
    if base is None:
        return None, "missing_base_module"
    return base.bleu_reward(
        pred,
        answer,
        target_lang=target_lang,
        tokenizer_mode=tokenizer_mode,
        spbleu_model_path="",
        return_meta=True,
    )


@lru_cache(maxsize=262_144)
def _cached_reference_length(reference_text: str, candidate_text: str, target_lang: str) -> tuple[Any, ...]:
    module = _dvao_module()
    base = getattr(module, "_BASE_MODULE", None)
    if base is None:
        return None, 0, 0, None, "missing_base_module"
    return base.reference_length_reward(reference_text, candidate_text, target_lang)


@lru_cache(maxsize=262_144)
def _cached_default_length(src_text: str, candidate_text: str) -> tuple[Any, ...]:
    module = _dvao_module()
    base = getattr(module, "_BASE_MODULE", None)
    if base is None:
        return -3.0, 0, 0, None
    return base.length_reward(src_text, candidate_text)


def _fast_local_components(item: dict[str, Any]) -> dict[str, Any]:
    """Fast path for local-only rewards: BLEU/spBLEU + length + must_gt.

    This intentionally bypasses the full verl ``compute_score`` wrapper when
    COMET/XCOMET/language/LLM rewards are disabled. It keeps the same component
    fields consumed by the static/DVAO aggregation code.
    """

    module = _dvao_module()
    base = getattr(module, "_BASE_MODULE", None)
    weights = _component_weights()
    data_source = str(item.get("data_source", "") or "")
    solution_str = str(item.get("completion", "") or "")
    ground_truth = str(item.get("ground_truth", "") or "")
    extra_info = dict(item.get("extra_info") or {})
    must_gt_input_path = _env("UNSLOTH_MT_MUST_GT_INPUT_PATH", _env("MUST_GT_INPUT_PATH", ""))

    src_lang = ""
    tgt_lang = str(extra_info.get("tgt_lang", "") or "")
    src_text = ""
    field_error = ""
    if base is not None:
        try:
            src_lang, tgt_lang, src_text = base._extract_commet_fields(extra_info)
        except Exception as exc:
            field_error = str(exc)
            src_text = str(
                extra_info.get("src_text")
                or extra_info.get("source_text")
                or extra_info.get("source")
                or extra_info.get("raw_source")
                or extra_info.get("input")
                or extra_info.get("query")
                or ""
            )
            tgt_lang = str(extra_info.get("tgt_lang", tgt_lang) or "")

    use_spbleu_component = bool(module._uses_spbleu_for_target(extra_info, solution_str, ground_truth))
    tokenizer_mode = "spbleu" if use_spbleu_component else "word"
    bleu_score, bleu_tokenizer = _cached_bleu_score(solution_str, ground_truth, tgt_lang, tokenizer_mode)
    bleu_available = bleu_score is not None and not use_spbleu_component
    spbleu_available = bleu_score is not None and use_spbleu_component
    bleu_norm = _clip01(bleu_score) if bleu_available else 0.0
    spbleu_norm = _clip01(bleu_score) if spbleu_available else 0.0

    default_raw_len_score, src_len, default_tgt_len, default_length_ratio = _cached_default_length(src_text, solution_str)
    raw_len_score = default_raw_len_score
    tgt_len = default_tgt_len
    length_ratio = default_length_ratio
    len_score = None
    length_reference_len = 0
    length_metric = "word_default"
    length_compare_target = "none"
    length_reward_min = -3.0
    length_reward_max = 1.0

    normalize_text = getattr(base, "_normalize_text", lambda value: " ".join(str(value or "").split()))
    normalize_match_text = getattr(base, "_normalize_match_text", lambda value: " ".join(str(value or "").lower().split()))
    if normalize_text(ground_truth):
        (
            reference_len_score,
            length_reference_len,
            reference_tgt_len,
            reference_length_ratio,
            reference_length_metric,
        ) = _cached_reference_length(ground_truth, solution_str, tgt_lang)
        if reference_len_score is not None:
            raw_len_score = reference_len_score
            len_score = reference_len_score
            tgt_len = reference_tgt_len
            length_ratio = reference_length_ratio
            length_metric = reference_length_metric
            length_compare_target = "reference"
            length_reward_min = -0.6
            length_reward_max = 0.0

    length_available = len_score is not None
    length_base_norm = (
        _normalize_linear(len_score, length_reward_min, length_reward_max)
        if length_available
        else 0.0
    )
    length_gamma = max(0.1, _env_float("UNSLOTH_MT_LENGTH_NORM_GAMMA", _env_float("LENGTH_NORM_GAMMA", 2.0)))
    length_base_norm = _clip01(length_base_norm ** length_gamma) if length_available else 0.0
    repetition_score, repetition_match, repetition_match_len = module._repetition_quality_score(solution_str)
    if length_available:
        length_norm = min(length_base_norm, repetition_score)
    elif repetition_score < 1.0:
        length_available = True
        length_norm = repetition_score
    else:
        length_norm = 0.0

    is_must_gt_input = getattr(base, "_is_must_gt_input", lambda *_args, **_kwargs: False)
    must_gt_input_matched = bool(is_must_gt_input(src_text, must_gt_input_path))
    must_gt_reference_matched = (
        must_gt_input_matched
        and bool(normalize_match_text(ground_truth))
        and normalize_match_text(solution_str) == normalize_match_text(ground_truth)
    )

    active_bleu_weight = weights["bleu"] if bleu_available else 0.0
    active_spbleu_weight = weights["spbleu"] if spbleu_available else 0.0
    active_length_weight = weights["length"] if length_available else 0.0
    fallback_score = _weighted_available_score(
        (
            (bleu_norm, bleu_available, active_bleu_weight),
            (spbleu_norm, spbleu_available, active_spbleu_weight),
            (length_norm, length_available, active_length_weight),
        )
    )

    return {
        "score": float(fallback_score),
        "quality_score": float(fallback_score),
        "base_score": float(fallback_score),
        "active_reward_weight": float(active_bleu_weight + active_spbleu_weight + active_length_weight),
        "data_source": data_source,
        "field_error": field_error,
        "src_lang": str(src_lang),
        "tgt_lang": str(tgt_lang),
        "src_text": src_text,
        "must_gt_input_matched": 1.0 if must_gt_input_matched else 0.0,
        "must_gt_reference_matched": 1.0 if must_gt_reference_matched else 0.0,
        "must_gt_reward_bonus": 0.0,
        "must_gt_fixed_bonus_disabled": 0.0,
        "must_gt_input_path": str(must_gt_input_path or ""),
        "bleu_score": float(bleu_score) if bleu_score is not None else -1.0,
        "bleu_available": 1.0 if bleu_available else 0.0,
        "bleu_tokenizer": str(bleu_tokenizer),
        "bleu_tokenizer_mode": tokenizer_mode,
        "spbleu_score": float(bleu_score) if spbleu_available else -1.0,
        "spbleu_available": 1.0 if spbleu_available else 0.0,
        "spbleu_tokenizer": str(bleu_tokenizer) if spbleu_available else "",
        "spbleu_triggered": 1.0 if use_spbleu_component else 0.0,
        "commet_raw_score": -1.0,
        "commet_score": -1.0,
        "commet_available": 0.0,
        "commet_unavailable": 0.0,
        "xcomet_raw_score": -1.0,
        "xcomet_score": -1.0,
        "xcomet_available": 0.0,
        "xcomet_unavailable": 0.0,
        "length_score": float(len_score) if len_score is not None else -1.0,
        "length_available": 1.0 if length_available else 0.0,
        "raw_length_score": float(raw_len_score),
        "source_len": float(src_len),
        "target_len": float(tgt_len),
        "length_ratio": float(length_ratio) if length_ratio is not None else -1.0,
        "length_reference_len": float(length_reference_len),
        "length_metric": str(length_metric),
        "length_compare_target": str(length_compare_target),
        "length_reward_min": float(length_reward_min),
        "length_reward_max": float(length_reward_max),
        "dvao_score": float(fallback_score),
        "dvao_bleu_score": float(bleu_norm),
        "dvao_bleu_available": 1.0 if bleu_available else 0.0,
        "dvao_spbleu_score": float(spbleu_norm),
        "dvao_spbleu_available": 1.0 if spbleu_available else 0.0,
        "dvao_commet_score": 0.0,
        "dvao_commet_available": 0.0,
        "dvao_xcomet_score": 0.0,
        "dvao_xcomet_available": 0.0,
        "dvao_length_score": float(length_norm),
        "dvao_length_available": 1.0 if length_available else 0.0,
        "dvao_length_base_score": float(length_base_norm),
        "dvao_length_norm_gamma": float(length_gamma),
        "dvao_length_repetition_score": float(repetition_score),
        "dvao_length_repetition_penalty": 1.0 if repetition_score < 1.0 else 0.0,
        "dvao_length_repetition_match_len": float(repetition_match_len),
        "dvao_length_repetition_match": str(repetition_match),
        "dvao_language_available": 0.0,
        "dvao_llm_available": 0.0,
        "dvao_static_bleu_weight": float(active_bleu_weight),
        "dvao_static_spbleu_weight": float(active_spbleu_weight),
        "dvao_static_commet_weight": 0.0,
        "dvao_static_xcomet_weight": 0.0,
        "dvao_static_length_weight": float(active_length_weight),
        "language_disabled_by_unsloth_adapter": 1.0,
        "fast_local_reward": 1.0,
    }


def _compute_components(
    *,
    sample_index: int = 0,
    data_source: str,
    completion: str,
    ground_truth: str,
    extra_info: dict[str, Any],
    include_xcomet: bool = True,
) -> dict[str, Any]:
    module = _dvao_module()
    _configure_verl_switches(module, include_xcomet=include_xcomet)
    weights = _component_weights()
    comet_url = _env("UNSLOTH_MT_COMET_URL", _env("RM_URL", ""))
    xcomet_url = _xcomet_url_for(sample_index) if include_xcomet else ""
    must_gt_input_path = _env("UNSLOTH_MT_MUST_GT_INPUT_PATH", _env("MUST_GT_INPUT_PATH", ""))

    kwargs: dict[str, Any] = {
        "data_source": data_source,
        "solution_str": completion,
        "ground_truth": ground_truth,
        "extra_info": extra_info,
        "bleu_weight": weights["bleu"],
        "spbleu_weight": weights["spbleu"],
        "commet_weight": weights["commet"],
        "xcomet_weight": weights["xcomet"],
        "length_weight": weights["length"],
        "language_weight": 0.0,
        "llm_weight": 0.0,
        "language_reward_enabled": False,
        "language_detect_url": "",
        "lang_detect_url": "",
        "commet_url": comet_url,
        "rm_url": comet_url,
        "xcomet_url": xcomet_url,
    }
    if must_gt_input_path:
        kwargs["must_gt_input_path"] = must_gt_input_path

    # Pass through common HTTP tuning knobs used by the verl reward files.
    for env_name, kw_name in {
        "UNSLOTH_MT_COMET_TIMEOUT": "commet_timeout",
        "UNSLOTH_MT_COMET_MAX_RETRIES": "commet_max_retries",
        "UNSLOTH_MT_COMET_RETRY_SLEEP": "commet_retry_sleep",
        "UNSLOTH_MT_COMET_BATCH_SIZE": "commet_batch_size",
        "UNSLOTH_MT_COMET_BATCH_WAIT_MS": "commet_batch_wait_ms",
        "UNSLOTH_MT_COMET_BATCH_QUEUE_SIZE": "commet_batch_queue_size",
        "UNSLOTH_MT_XCOMET_TIMEOUT": "xcomet_timeout",
        "UNSLOTH_MT_XCOMET_MAX_RETRIES": "xcomet_max_retries",
        "UNSLOTH_MT_XCOMET_RETRY_SLEEP": "xcomet_retry_sleep",
        "UNSLOTH_MT_XCOMET_BATCH_SIZE": "xcomet_batch_size",
        "UNSLOTH_MT_XCOMET_BATCH_WAIT_MS": "xcomet_batch_wait_ms",
        "UNSLOTH_MT_XCOMET_BATCH_QUEUE_SIZE": "xcomet_batch_queue_size",
        "UNSLOTH_MT_LENGTH_NORM_GAMMA": "length_norm_gamma",
        "UNSLOTH_MT_XCOMET_NORM_GAMMA": "xcomet_norm_gamma",
    }.items():
        value = _env(env_name, "")
        if value != "":
            kwargs[kw_name] = value

    try:
        result = module.compute_score(**kwargs)
        result = dict(result or {})
    except Exception as exc:
        result = {
            "score": 0.0,
            "quality_score": 0.0,
            "reward_error": repr(exc),
            "dvao_reward_enabled": 0.0,
        }

    # Force explicit static component weights for later aggregation. The verl
    # DVAO module zeros BLEU or spBLEU depending on target language; we keep that
    # behavior and only use available components.
    for name, weight in weights.items():
        result.setdefault(f"dvao_static_{name}_weight", float(weight))
    result.setdefault("dvao_language_available", 0.0)
    result.setdefault("dvao_llm_available", 0.0)
    result["language_disabled_by_unsloth_adapter"] = 1.0
    return result


def _attach_direct_xcomet_batch(records: list[dict[str, Any]], items: list[dict[str, Any]]) -> None:
    """Attach XCOMET components with true URL-level batched HTTP requests.

    The imported verl XCOMET module already knows the service payload/response
    schema. This adapter-level path avoids the older pattern where many Python
    worker threads each submit one sample and rely on an internal queue to
    opportunistically form small batches.
    """

    urls = _xcomet_urls()
    if not urls or not records:
        return

    module = _dvao_module()
    xmod = getattr(module, "_XCOMET_MODULE", None)
    if xmod is None:
        return

    timeout = _env_float("UNSLOTH_MT_XCOMET_TIMEOUT", _env_float("XCOMET_TIMEOUT", 30.0))
    max_retries = max(0, _env_int("UNSLOTH_MT_XCOMET_MAX_RETRIES", 0))
    retry_sleep = max(0.0, _env_float("UNSLOTH_MT_XCOMET_RETRY_SLEEP", 1.0))
    batch_size = max(1, _env_int("UNSLOTH_MT_XCOMET_BATCH_SIZE", 1024))
    gamma = max(0.1, _env_float("UNSLOTH_MT_XCOMET_NORM_GAMMA", 2.0))
    empty_raw = _env_float("UNSLOTH_MT_EMPTY_TARGET_XCOMET_RAW_SCORE", 0.1)

    def mark_unavailable(idx: int, error: str = "") -> None:
        record = records[idx]
        record["dvao_xcomet_score"] = 0.0
        record["dvao_xcomet_available"] = 0.0
        record["xcomet_raw_score"] = -1.0
        record["xcomet_score"] = -1.0
        record["xcomet_score_mode"] = ""
        record["xcomet_available"] = 0.0
        record["xcomet_unavailable"] = 1.0
        record["xcomet_error"] = str(error)

    def attach_raw(idx: int, raw_score: float, selected_url: str, mode: str = "raw") -> None:
        raw_score = _safe_float(raw_score, -1.0)
        norm_score = _clip01(_clip01(raw_score) ** gamma)
        record = records[idx]
        record["dvao_xcomet_score"] = float(norm_score)
        record["dvao_xcomet_available"] = 1.0
        record["xcomet_raw_score"] = float(raw_score)
        record["xcomet_score"] = float(raw_score)
        record["xcomet_score_mode"] = mode
        record["xcomet_norm_gamma"] = float(gamma)
        record["xcomet_available"] = 1.0
        record["xcomet_unavailable"] = 0.0
        record["xcomet_error"] = ""
        record["xcomet_selected_url"] = selected_url

    pending_by_url: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, item in enumerate(items):
        completion = str(item.get("completion", "") or "")
        ground_truth = str(item.get("ground_truth", "") or "")
        extra_info = dict(item.get("extra_info") or {})
        data_source = str(item.get("data_source", "") or "")
        selected_url = _xcomet_url_for(idx)
        records[idx]["xcomet_selected_url"] = selected_url

        try:
            src_lang, tgt_lang, src_text = xmod._extract_commet_fields(extra_info)
        except Exception as exc:
            mark_unavailable(idx, f"field_error={exc}")
            continue

        if not selected_url:
            mark_unavailable(idx, "missing_xcomet_url")
        elif src_text and not ground_truth.strip():
            mark_unavailable(idx, "empty_ground_truth")
        elif src_text and not completion.strip():
            attach_raw(idx, empty_raw, selected_url, mode="empty_target_default_raw")
        else:
            payload = xmod._build_commet_payload(
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_text=src_text,
                tgt_text=completion,
                ground_truth=ground_truth,
                data_source=data_source,
            )
            pending_by_url[selected_url].append((idx, payload))

    def call_chunk(url: str, chunk: list[tuple[int, dict[str, Any]]]) -> None:
        payloads = [payload for _, payload in chunk]
        last_error = ""
        for attempt in range(max_retries + 1):
            try:
                raw_scores = xmod._call_commet_model_batch(url, payloads, timeout)
                for (idx, _), raw_score in zip(chunk, raw_scores, strict=True):
                    attach_raw(idx, raw_score, url)
                return
            except Exception as exc:
                last_error = repr(exc)
                if attempt < max_retries and retry_sleep > 0:
                    time.sleep(retry_sleep)
        for idx, _ in chunk:
            mark_unavailable(idx, last_error)

    max_url_workers = max(1, min(len(pending_by_url), _env_int("UNSLOTH_MT_XCOMET_URL_WORKERS", len(urls))))
    jobs: list[tuple[str, list[tuple[int, dict[str, Any]]]]] = []
    for url, pending in pending_by_url.items():
        for start in range(0, len(pending), batch_size):
            jobs.append((url, pending[start : start + batch_size]))

    if len(jobs) <= 1:
        for url, chunk in jobs:
            call_chunk(url, chunk)
    else:
        with ThreadPoolExecutor(max_workers=max_url_workers, thread_name_prefix="unsloth-xcomet-batch") as pool:
            list(pool.map(lambda job: call_chunk(job[0], job[1]), jobs))


def _static_score(record: dict[str, Any]) -> float:
    weights = _component_weights()
    return _weighted_available_score(
        [
            _available_component(record, "bleu", weights["bleu"]),
            _available_component(record, "spbleu", weights["spbleu"]),
            _available_component(record, "length", weights["length"]),
            _available_component(record, "commet", weights["commet"]),
            _available_component(record, "xcomet", weights["xcomet"]),
        ]
    )


def _group_indices(prompts: list[Any], num_generations: int | None) -> list[list[int]]:
    n = len(prompts)
    if num_generations and num_generations > 1 and n % num_generations == 0:
        return [list(range(i, min(i + num_generations, n))) for i in range(0, n, num_generations)]

    groups: dict[str, list[int]] = defaultdict(list)
    order: list[str] = []
    for idx, prompt in enumerate(prompts):
        key = _stable_prompt_key(prompt)
        if key not in groups:
            order.append(key)
        groups[key].append(idx)
    return [groups[key] for key in order]


def _apply_group_normalization(scores: list[float], prompts: list[Any]) -> list[float]:
    num_generations = _env_int("UNSLOTH_MT_NUM_GENERATIONS", 0)
    normalized = list(scores)
    for indices in _group_indices(prompts, num_generations if num_generations > 1 else None):
        if len(indices) < 2:
            continue
        vals = [scores[i] for i in indices]
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / len(vals)
        std = math.sqrt(var)
        if std <= 1e-8:
            for i in indices:
                normalized[i] = 0.0
        else:
            for i in indices:
                normalized[i] = (scores[i] - mean) / (std + 1e-4)
    return normalized


def _apply_must_gt_anchor(records: list[dict[str, Any]], scores: list[float], prompts: list[Any]) -> list[float]:
    margin = _env_float("UNSLOTH_MT_MUST_GT_DYNAMIC_MARGIN", _env_float("MUST_GT_DYNAMIC_MARGIN", 0.1))
    num_generations = _env_int("UNSLOTH_MT_NUM_GENERATIONS", 0)
    adjusted = list(scores)
    anchors = 0
    changed = 0
    for indices in _group_indices(prompts, num_generations if num_generations > 1 else None):
        if not indices:
            continue
        group_max = max(adjusted[i] for i in indices)
        for i in indices:
            if _safe_float(records[i].get("must_gt_reference_matched", 0.0), 0.0) <= 0.5:
                continue
            anchors += 1
            target = max(adjusted[i], group_max + margin)
            if target > adjusted[i]:
                adjusted[i] = target
                changed += 1
            records[i]["must_gt_dynamic_anchor_applied"] = 1.0 if target > scores[i] else 0.0
            records[i]["must_gt_dynamic_anchor_group_max"] = float(group_max)
            records[i]["must_gt_dynamic_anchor_target_score"] = float(target)
            records[i]["must_gt_dynamic_anchor_margin"] = float(margin)
    if anchors:
        print(f"[Unsloth-MT-Reward][must_gt] anchors={anchors} adjusted={changed} margin={margin}")
    return adjusted


def _dvao_weight_caps(names: list[str]) -> tuple[list[float], list[float]]:
    defaults = {
        "bleu": 0.15,
        "spbleu": 0.15,
        "commet": 0.15,
        "xcomet": 0.15,
        "length": 0.15,
    }
    caps = []
    mins = []
    for name in names:
        raw_cap = _env(f"UNSLOTH_MT_DVAO_MAX_WEIGHT_{name.upper()}", _env(f"DVAO_MAX_WEIGHT_{name.upper()}", ""))
        if raw_cap.lower() in {"none", "null", "inf", "infinite", "unlimited"}:
            cap = float("inf")
        elif raw_cap:
            cap = max(0.0, _safe_float(raw_cap, defaults.get(name, 1.0)))
        else:
            cap = defaults.get(name, 1.0)
        caps.append(cap)
        raw_min = _env(f"UNSLOTH_MT_DVAO_MIN_WEIGHT_{name.upper()}", _env(f"DVAO_MIN_WEIGHT_{name.upper()}", ""))
        mins.append(max(0.0, _safe_float(raw_min, 0.0)))
    return caps, mins


def _cap_weights(weights: list[float], available: list[bool], names: list[str]) -> list[float]:
    eps = 1e-8
    caps, mins = _dvao_weight_caps(names)
    w = [max(0.0, x) if ok else 0.0 for x, ok in zip(weights, available, strict=True)]
    if sum(w) <= eps:
        return w
    w = [x / sum(w) for x in w]
    w = [min(x, cap) for x, cap in zip(w, caps, strict=True)]
    residual = 1.0 - sum(w)
    if residual > eps:
        room = [max(0.0, cap - x) if ok and math.isfinite(cap) else (float("inf") if ok else 0.0)
                for x, cap, ok in zip(w, caps, available, strict=True)]
        eligible = [ok and r > eps for ok, r in zip(available, room, strict=True)]
        if any(eligible):
            base = [weights[i] if eligible[i] else 0.0 for i in range(len(w))]
            if sum(base) <= eps:
                base = [1.0 if eligible[i] else 0.0 for i in range(len(w))]
            for i in range(len(w)):
                if eligible[i]:
                    add = residual * base[i] / sum(base)
                    if math.isfinite(room[i]):
                        add = min(add, room[i])
                    w[i] += add
    min_w = [min(mn, cap) if ok else 0.0 for mn, cap, ok in zip(mins, caps, available, strict=True)]
    if sum(min_w) > eps:
        if sum(min_w) >= 1.0 - eps:
            return [x / sum(min_w) for x in min_w]
        w = [max(x, mn) for x, mn in zip(w, min_w, strict=True)]
        if sum(w) > 1.0 + eps:
            excess = sum(w) - 1.0
            reducible = [max(0.0, x - mn) for x, mn in zip(w, min_w, strict=True)]
            if sum(reducible) > eps:
                w = [x - excess * r / sum(reducible) for x, r in zip(w, reducible, strict=True)]
    if sum(w) > eps:
        w = [x / sum(w) for x in w]
    return w


def _apply_dvao(records: list[dict[str, Any]], prompts: list[Any]) -> list[float]:
    component_names = ["bleu", "spbleu", "length", "commet", "xcomet"]
    static_weights = _component_weights()
    eps = _env_float("UNSLOTH_MT_DVAO_VARIANCE_EPS", _env_float("DVAO_VARIANCE_EPS", 1e-6))
    min_variance = _env_float("UNSLOTH_MT_DVAO_MIN_VARIANCE", _env_float("DVAO_MIN_VARIANCE", 0.0))
    num_generations = _env_int("UNSLOTH_MT_NUM_GENERATIONS", 0)

    scores = [_static_score(record) for record in records]
    for indices in _group_indices(prompts, num_generations if num_generations > 1 else None):
        if not indices:
            continue
        variances: list[float] = []
        group_available: list[bool] = []
        values_by_component: list[list[float]] = []
        available_by_component: list[list[bool]] = []

        for name in component_names:
            vals = []
            avails = []
            for idx in indices:
                score, available, _ = _available_component(records[idx], name, static_weights[name])
                vals.append(score)
                avails.append(available)
            valid_vals = [v for v, ok in zip(vals, avails, strict=True) if ok]
            group_available.append(bool(valid_vals))
            if len(valid_vals) >= 2:
                mean = sum(valid_vals) / len(valid_vals)
                variance = sum((x - mean) ** 2 for x in valid_vals) / len(valid_vals)
            else:
                variance = 0.0
            variances.append(max(variance, min_variance) if valid_vals else 0.0)
            values_by_component.append(vals)
            available_by_component.append(avails)

        weighted_variances = [
            variances[i] * static_weights[component_names[i]] if group_available[i] else 0.0
            for i in range(len(component_names))
        ]
        if sum(weighted_variances) <= eps:
            group_weights = [
                static_weights[name] if group_available[i] else 0.0
                for i, name in enumerate(component_names)
            ]
        else:
            group_weights = weighted_variances
        if sum(group_weights) <= eps:
            group_weights = [1.0 if ok else 0.0 for ok in group_available]
        group_weights = _cap_weights(group_weights, group_available, component_names)

        for local_pos, sample_idx in enumerate(indices):
            sample_available = [available_by_component[i][local_pos] for i in range(len(component_names))]
            sample_weights = [group_weights[i] if sample_available[i] else 0.0 for i in range(len(component_names))]
            if sum(sample_weights) <= eps:
                sample_weights = [
                    static_weights[name] if sample_available[i] else 0.0
                    for i, name in enumerate(component_names)
                ]
            sample_weights = _cap_weights(sample_weights, sample_available, component_names)
            if sum(sample_weights) <= eps:
                continue
            values = [values_by_component[i][local_pos] for i in range(len(component_names))]
            scores[sample_idx] = sum(w * v for w, v in zip(sample_weights, values, strict=True))
            records[sample_idx]["dvao_score"] = float(scores[sample_idx])
            for i, name in enumerate(component_names):
                records[sample_idx][f"dvao_{name}_weight"] = float(sample_weights[i])
                records[sample_idx][f"dvao_{name}_variance"] = float(variances[i])

    if records:
        print(
            "[Unsloth-MT-Reward][DVAO] "
            f"samples={len(records)} groups={len(_group_indices(prompts, num_generations if num_generations > 1 else None))} "
            f"mean_score={sum(scores) / len(scores):.6f}"
        )
    return scores


def _mean_available(records: list[dict[str, Any]], score_key: str, available_key: str) -> str:
    vals = [
        _safe_float(record.get(score_key, 0.0), 0.0)
        for record in records
        if _safe_float(record.get(available_key, 0.0), 0.0) > 0.5
    ]
    if not vals:
        return "n/a"
    return f"{sum(vals) / len(vals):.6f}"


def _mean_available_float(records: list[dict[str, Any]], score_key: str, available_key: str) -> float | None:
    vals = [
        _safe_float(record.get(score_key, 0.0), 0.0)
        for record in records
        if _safe_float(record.get(available_key, 0.0), 0.0) > 0.5
    ]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _short_text(value: Any, max_chars: int = 260) -> str:
    text = " ".join(str(value or "").strip().split())
    return text if len(text) <= max_chars else text[:max_chars] + "...(truncated)"


def _source_text_from_record(record: dict[str, Any], item: dict[str, Any]) -> str:
    src_text = record.get("src_text", "")
    if src_text:
        return str(src_text)
    extra_info = item.get("extra_info", {}) if isinstance(item, dict) else {}
    if isinstance(extra_info, dict):
        for key in ("src_text", "source_text", "source", "raw_source", "input", "query"):
            value = extra_info.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


def _print_sample_logs(
    version: str,
    records: list[dict[str, Any]],
    scores: list[float],
    items: list[dict[str, Any]],
    trainer_state: Any = None,
) -> None:
    sample_count = max(0, _env_int("UNSLOTH_MT_SAMPLE_LOG_COUNT", 0))
    if sample_count <= 0 or not records:
        return
    step = getattr(trainer_state, "global_step", None)
    for rank, idx in enumerate(range(min(sample_count, len(records))), start=1):
        record = records[idx]
        item = items[idx] if idx < len(items) else {}
        print(
            "\n"
            + "=" * 96
            + "\n"
            + f"[Unsloth-MT-Sample] step={step} version={version} sample={rank}/{sample_count} idx={idx}\n"
            + f"src={_short_text(_source_text_from_record(record, item))}\n"
            + f"pred={_short_text(item.get('completion', ''))}\n"
            + f"gt={_short_text(item.get('ground_truth', ''))}\n"
            + "reward_components="
            + json.dumps(
                {
                    "total_reward": float(scores[idx]) if idx < len(scores) else 0.0,
                    "bleu": _safe_float(record.get("dvao_bleu_score", 0.0), 0.0),
                    "spbleu": _safe_float(record.get("dvao_spbleu_score", 0.0), 0.0),
                    "length": _safe_float(record.get("dvao_length_score", 0.0), 0.0),
                    "commet": _safe_float(record.get("dvao_commet_score", 0.0), 0.0),
                    "xcomet": _safe_float(record.get("dvao_xcomet_score", 0.0), 0.0),
                    "must_gt_input": _safe_float(record.get("must_gt_input_matched", 0.0), 0.0),
                    "must_gt_ref": _safe_float(record.get("must_gt_reference_matched", 0.0), 0.0),
                },
                ensure_ascii=False,
            )
            + "\n"
            + "=" * 96,
            flush=True,
        )


def _wandb_log_summary(
    payload: dict[str, Any],
    records: list[dict[str, Any]],
    scores: list[float],
    items: list[dict[str, Any]],
    trainer_state: Any = None,
) -> None:
    if not _env_bool("UNSLOTH_MT_WANDB_REWARD_LOG", False):
        return
    try:
        import wandb

        run = getattr(wandb, "run", None)
        if run is None:
            return
        step = getattr(trainer_state, "global_step", None)
        metrics: dict[str, float] = {
            "mt_reward/total_mean": float(payload.get("score_mean", 0.0)),
            "mt_reward/total_min": float(payload.get("score_min", 0.0)),
            "mt_reward/total_max": float(payload.get("score_max", 0.0)),
            "mt_reward/must_gt_input_count": float(payload.get("must_gt_input", 0)),
            "mt_reward/must_gt_ref_count": float(payload.get("must_gt_ref", 0)),
            "mt_reward/samples": float(payload.get("samples", 0)),
        }
        for name in ("bleu", "spbleu", "length", "commet", "xcomet"):
            value = _mean_available_float(records, f"dvao_{name}_score", f"dvao_{name}_available")
            if value is not None:
                metrics[f"mt_reward/{name}_mean"] = float(value)
            available_count = sum(
                1
                for record in records
                if _safe_float(record.get(f"dvao_{name}_available", 0.0), 0.0) > 0.5
            )
            metrics[f"mt_reward/{name}_available_count"] = float(available_count)
        wandb.log(metrics)

        sample_count = max(0, _env_int("UNSLOTH_MT_WANDB_SAMPLE_COUNT", 0))
        if sample_count > 0 and records:
            rows = []
            for idx in range(min(sample_count, len(records))):
                record = records[idx]
                item = items[idx] if idx < len(items) else {}
                rows.append(
                    [
                        step,
                        idx,
                        _short_text(_source_text_from_record(record, item), 500),
                        _short_text(item.get("completion", ""), 500),
                        _short_text(item.get("ground_truth", ""), 500),
                        float(scores[idx]) if idx < len(scores) else 0.0,
                        _safe_float(record.get("dvao_bleu_score", 0.0), 0.0),
                        _safe_float(record.get("dvao_spbleu_score", 0.0), 0.0),
                        _safe_float(record.get("dvao_length_score", 0.0), 0.0),
                        _safe_float(record.get("dvao_commet_score", 0.0), 0.0),
                        _safe_float(record.get("dvao_xcomet_score", 0.0), 0.0),
                    ]
                )
            table = wandb.Table(
                columns=[
                    "step",
                    "idx",
                    "src",
                    "pred",
                    "gt",
                    "total_reward",
                    "bleu",
                    "spbleu",
                    "length",
                    "commet",
                    "xcomet",
                ],
                data=rows,
            )
            wandb.log({"mt_reward/samples_table": table})
    except Exception as exc:
        if _env_bool("UNSLOTH_MT_WANDB_REWARD_LOG_ERRORS", False):
            print(f"[Unsloth-MT-Reward][wandb_log_error] {exc!r}", flush=True)


def _log_summary(version: str, records: list[dict[str, Any]], scores: list[float], trainer_state: Any = None) -> dict[str, Any]:
    if not _env_bool("UNSLOTH_MT_REWARD_LOG_ENABLED", True):
        return {}
    log_path = Path(_env("UNSLOTH_MT_REWARD_LOG_PATH", str(DEFAULT_LOG_PATH)))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    step = getattr(trainer_state, "global_step", None)
    payload = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": version,
        "step": step,
        "samples": len(records),
        "score_mean": float(sum(scores) / len(scores)) if scores else 0.0,
        "score_min": float(min(scores)) if scores else 0.0,
        "score_max": float(max(scores)) if scores else 0.0,
        "bleu_mean": _mean_available(records, "dvao_bleu_score", "dvao_bleu_available"),
        "spbleu_mean": _mean_available(records, "dvao_spbleu_score", "dvao_spbleu_available"),
        "length_mean": _mean_available(records, "dvao_length_score", "dvao_length_available"),
        "commet_mean": _mean_available(records, "dvao_commet_score", "dvao_commet_available"),
        "xcomet_mean": _mean_available(records, "dvao_xcomet_score", "dvao_xcomet_available"),
        "must_gt_input": sum(1 for r in records if _safe_float(r.get("must_gt_input_matched", 0.0), 0.0) > 0.5),
        "must_gt_ref": sum(1 for r in records if _safe_float(r.get("must_gt_reference_matched", 0.0), 0.0) > 0.5),
        "language_disabled": True,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(
        "[Unsloth-MT-Reward] "
        f"version={version} step={step} samples={payload['samples']} "
        f"score_mean={payload['score_mean']:.6f} "
        f"bleu={payload['bleu_mean']} spbleu={payload['spbleu_mean']} "
        f"commet={payload['commet_mean']} xcomet={payload['xcomet_mean']} "
        f"must_gt_input={payload['must_gt_input']} must_gt_ref={payload['must_gt_ref']}"
    )
    return payload


def _reward_impl(version: str, prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
    reward_start = time.monotonic()
    n = len(completions)
    prompts = _as_list(prompts, n, default="")
    data_sources = _as_list(kwargs.get("data_source"), n, default="unsloth_mt")
    reward_models = _as_list(kwargs.get("reward_model"), n, default={})
    explicit_gt = _as_list(kwargs.get("ground_truth"), n, default=None)
    extra_infos = [_dict_or_empty(x) for x in _as_list(kwargs.get("extra_info"), n, default={})]
    trainer_state = kwargs.get("trainer_state")

    def build_item(i: int, completion_obj: Any) -> dict[str, Any]:
        completion = _completion_to_text(completion_obj)
        ground_truth = _ground_truth_for(i, reward_models, extra_infos, explicit_gt)
        return {
            "sample_index": i,
            "data_source": str(data_sources[i]),
            "completion": completion,
            "ground_truth": ground_truth,
            "extra_info": extra_infos[i],
        }

    items = [build_item(i, completion_obj) for i, completion_obj in enumerate(completions)]
    fast_local_reward = _env_bool("UNSLOTH_MT_FAST_LOCAL_REWARD", False)
    direct_xcomet_batch = (
        not fast_local_reward
        and _env_bool("UNSLOTH_MT_DIRECT_XCOMET_BATCH", False)
        and bool(_xcomet_urls())
    )

    def build_one(item: dict[str, Any]) -> dict[str, Any]:
        if fast_local_reward:
            return _fast_local_components(item)
        return _compute_components(
            sample_index=int(item["sample_index"]),
            data_source=str(item["data_source"]),
            completion=str(item["completion"]),
            ground_truth=str(item["ground_truth"]),
            extra_info=dict(item["extra_info"]),
            include_xcomet=not direct_xcomet_batch,
        )

    workers = max(1, _env_int("UNSLOTH_MT_REWARD_WORKERS", min(64, max(1, n))))
    if workers > 1 and n > 1:
        workers = min(workers, n)
        module = _dvao_module()  # load once before worker threads touch the module cache
        _prewarm_verl_caches(module)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="unsloth-mt-reward") as pool:
            records = list(pool.map(build_one, items))
    else:
        records = [build_one(item) for item in items]
    component_elapsed = time.monotonic() - reward_start

    if direct_xcomet_batch:
        _attach_direct_xcomet_batch(records, items)
    xcomet_elapsed = time.monotonic() - reward_start - component_elapsed

    if version == "v3_dvao":
        scores = _apply_dvao(records, prompts)
        scores = _apply_must_gt_anchor(records, scores, prompts)
    else:
        scores = [_static_score(record) for record in records]
        if version == "v2_mustgt":
            scores = _apply_must_gt_anchor(records, scores, prompts)
            if _env_bool("UNSLOTH_MT_GROUP_NORMALIZE", False):
                scores = _apply_group_normalization(scores, prompts)

    summary_payload = _log_summary(version, records, scores, trainer_state)
    _print_sample_logs(version, records, scores, items, trainer_state)
    _wandb_log_summary(summary_payload, records, scores, items, trainer_state)
    total_elapsed = time.monotonic() - reward_start
    print(
        "[Unsloth-MT-Reward-Time] "
        f"samples={n} workers={workers} fast_local={int(fast_local_reward)} "
        f"components_sec={component_elapsed:.2f} "
        f"xcomet_sec={xcomet_elapsed:.2f} "
        f"total_sec={total_elapsed:.2f}",
        flush=True,
    )
    return [float(x) for x in scores]


def mt_reward_static_v1(prompts, completions, completion_ids=None, **kwargs) -> list[float]:
    """First migration version: static component fusion, language disabled."""

    return _reward_impl("v1_static", prompts=list(prompts), completions=list(completions), **kwargs)


def mt_reward_mustgt_v2(prompts, completions, completion_ids=None, **kwargs) -> list[float]:
    """Second migration version: v1 + must_gt anchor/logging/optional group norm."""

    return _reward_impl("v2_mustgt", prompts=list(prompts), completions=list(completions), **kwargs)


def mt_reward_dvao_v3(prompts, completions, completion_ids=None, **kwargs) -> list[float]:
    """Third migration version: approximate verl DVAO inside TRL reward_func."""

    return _reward_impl("v3_dvao", prompts=list(prompts), completions=list(completions), **kwargs)


def get_reward_func(version: str):
    version = str(version or "v1").strip().lower()
    if version in {"1", "v1", "static", "v1_static"}:
        return mt_reward_static_v1
    if version in {"2", "v2", "mustgt", "must_gt", "v2_mustgt"}:
        return mt_reward_mustgt_v2
    if version in {"3", "v3", "dvao", "v3_dvao"}:
        return mt_reward_dvao_v3
    raise ValueError(f"Unknown reward version: {version}")
