"""Pilot LoRA fine-tuning of StyleTTS2's prosody diffusion branch.

This adapts only the diffusion U-Net inside ``model.diffusion`` with manual
LoRA.  The text encoder, BERT, style encoder, predictor, decoder, CFVN, CVAE,
and style adapter are frozen.  It is a research pilot: it can measure WER,
speaker similarity, F0, duration, and energy, but it cannot establish
naturalness or MOS without real listener ratings.

Typical run::

    python scripts/training/train_styletts2_lora.py \
        --manifest data/processed/styletts2_lora/manifest.csv \
        --epochs 10 --eval-samples 25

Use ``--eval-only`` to re-run the paired base-vs-LoRA evaluation from a saved
adapter without training again.  Use ``--max-steps`` for a smoke run.  On a
low-RAM machine, combine ``--eval-only`` with ``--eval-generate-only`` and score
the wavs with ``scripts/evaluation/evaluate_n1_metrics.py`` in bounded slices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from voix.training.lora import (
    adapter_statistics,
    count_trainable_parameters,
    inject_lora,
    load_lora,
    save_lora,
    set_lora_enabled,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "styletts2_lora" / "manifest.csv"
DEFAULT_OUTPUT = ROOT / "results" / "checkpoints" / "styletts2_lora_prosody.pt"
DEFAULT_EVAL_DIR = ROOT / "results" / "styletts2_lora_eval"
REQUIRED_COLUMNS = {
    "clip_id",
    "speaker_id",
    "split",
    "wav_path",
    "ref_clip_id",
    "ref_wav_path",
    "duration_sec",
    "text",
}
METRIC_NAMES = (
    "wer",
    "speaker_similarity_target",
    "speaker_similarity_ref",
    "f0_mean_hz",
    "f0_std_hz",
    "f0_range_hz",
    "f0_voiced_ratio",
    "duration_seconds",
    "energy_mean",
    "energy_std",
)


def resolve_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def load_manifest(path: Path) -> pd.DataFrame:
    """Load and validate a speaker-disjoint train/val manifest."""
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"manifest is missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"manifest is empty: {path}")
    for column in (
        "clip_id",
        "speaker_id",
        "split",
        "wav_path",
        "ref_wav_path",
        "text",
    ):
        if frame[column].isna().any():
            raise ValueError(f"manifest column {column!r} contains null values")

    frame = frame[frame["split"].isin(["train", "val"])].copy()
    if frame.empty:
        raise ValueError("manifest has no train/val rows")
    if not (frame["split"] == "train").any():
        raise ValueError("manifest has no train rows")
    if not (frame["split"] == "val").any():
        raise ValueError("manifest has no val rows")

    train_speakers = set(frame.loc[frame["split"] == "train", "speaker_id"].astype(str))
    val_speakers = set(frame.loc[frame["split"] == "val", "speaker_id"].astype(str))
    overlap = sorted(train_speakers & val_speakers)
    if overlap:
        raise ValueError(
            "train/val speakers overlap; the split is not speaker-disjoint: "
            f"{overlap[:5]}"
        )

    for column in ("wav_path", "ref_wav_path"):
        frame[column] = frame[column].map(lambda value: str(resolve_path(value)))
    missing_files = sorted(
        {
            value
            for column in ("wav_path", "ref_wav_path")
            for value in frame[column]
            if not Path(value).is_file()
        }
    )
    if missing_files:
        raise FileNotFoundError(
            f"manifest references {len(missing_files)} missing wav files; "
            f"first: {missing_files[0]}"
        )
    frame["duration_sec"] = pd.to_numeric(frame["duration_sec"], errors="coerce")
    if frame["duration_sec"].isna().any():
        raise ValueError("manifest contains non-numeric duration_sec values")
    return frame


def manifest_digest(frame: pd.DataFrame) -> str:
    payload = "|".join(sorted(frame["clip_id"].astype(str)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def warmup_cosine(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_ratio: float = 0.05,
) -> float:
    """Linear warmup followed by cosine decay, both in [min_ratio, 1]."""
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    min_ratio = float(np.clip(min_ratio, 0.0, 1.0))
    step = max(0, int(step))
    if warmup_steps > 0 and step < warmup_steps:
        return float((step + 1) / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = float(np.clip(progress, 0.0, 1.0))
    return float(
        min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    )


def select_eval_records(
    frame: pd.DataFrame,
    limit: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Deterministically pick held-out-speaker evaluation clips."""
    if limit <= 0:
        return []
    val = frame[frame["split"] == "val"].copy()
    if val.empty:
        return []
    val["_key"] = [
        hashlib.sha256(f"{seed}:{clip}".encode("utf-8")).hexdigest()
        for clip in val["clip_id"].astype(str)
    ]
    val = val.sort_values(["_key", "clip_id"]).head(int(limit))
    return val.drop(columns="_key").to_dict("records")


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean_std(values: Iterable[Any]) -> Optional[Dict[str, float]]:
    clean = [_finite(value) for value in values]
    clean = [value for value in clean if value is not None]
    if not clean:
        return None
    array = np.asarray(clean, dtype=np.float64)
    return {
        "mean": round(float(array.mean()), 6),
        "std": round(float(array.std()), 6),
        "n": int(array.size),
    }


def summarize_generation(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate paired base/LoRA generation rows into honest summary stats."""
    if not rows:
        return {"status": "unavailable", "reason": "no generation rows", "n": 0}
    conditions: Dict[str, Any] = {}
    for condition in ("base", "lora"):
        subset = [row for row in rows if row.get("condition") == condition]
        metrics: Dict[str, Any] = {}
        for name in METRIC_NAMES:
            metrics[name] = _mean_std(row.get(name) for row in subset)
        conditions[condition] = {"n": len(subset), "metrics": metrics}

    delta: Dict[str, Optional[float]] = {}
    for name in METRIC_NAMES:
        base = conditions["base"]["metrics"].get(name)
        lora = conditions["lora"]["metrics"].get(name)
        delta[name] = (
            round(float(lora["mean"] - base["mean"]), 6)
            if base is not None and lora is not None
            else None
        )

    # The same clips are generated under both conditions, so report the
    # within-clip paired difference as well as the difference of means.
    by_clip: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_clip.setdefault(str(row.get("clip_id")), {})[str(row.get("condition"))] = row
    paired: Dict[str, Any] = {}
    for name in METRIC_NAMES:
        differences = []
        for pair in by_clip.values():
            if "base" not in pair or "lora" not in pair:
                continue
            base_value = _finite(pair["base"].get(name))
            lora_value = _finite(pair["lora"].get(name))
            if base_value is not None and lora_value is not None:
                differences.append(lora_value - base_value)
        paired[name] = _mean_std(differences)

    return {
        "status": "computed",
        "n": len(rows),
        "conditions": conditions,
        "delta_lora_minus_base": delta,
        "paired_delta_lora_minus_base": paired,
    }


def build_tokens(
    text: str,
    phoneme_converter,
    text_cleaner,
    max_tokens: int = 500,
) -> torch.Tensor:
    """Phonemize text into the BERT token sequence StyleTTS2 expects."""
    from nltk.tokenize import word_tokenize

    phonemes = phoneme_converter.phonemize(str(text))
    tokens = text_cleaner(" ".join(word_tokenize(phonemes)))
    tokens = [0] + list(tokens)
    if len(tokens) < 2:
        raise ValueError("empty phoneme sequence")
    if len(tokens) > max_tokens:
        raise ValueError(f"phoneme sequence too long for BERT ({len(tokens)} tokens)")
    return torch.tensor(tokens, dtype=torch.long)


def extract_style(model, wav_path: str, device: str) -> torch.Tensor:
    """Return the 256-D StyleTTS2 style/prosody vector for one wav."""
    import librosa
    from styletts2.tts import preprocess

    wave, _ = librosa.load(str(wav_path), sr=24000, mono=True)
    if wave.size < 24000 // 4:
        raise ValueError(f"reference audio too short: {wav_path}")
    mel = preprocess(wave).to(device)
    with torch.no_grad():
        style = model.style_encoder(mel.unsqueeze(1))
        prosody = model.predictor_encoder(mel.unsqueeze(1))
    return torch.cat([style, prosody], dim=-1)


def prepare_example(
    model,
    record: Dict[str, Any],
    tokenizer: Tuple[Any, Any],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the target style and BERT conditioning for one clip."""
    import librosa
    from styletts2.tts import length_to_mask, preprocess

    phoneme_converter, text_cleaner = tokenizer
    tokens = build_tokens(record["text"], phoneme_converter, text_cleaner)
    wave, _ = librosa.load(str(record["wav_path"]), sr=24000, mono=True)
    if wave.size < 24000 // 2:
        raise ValueError(f"target audio too short: {record['wav_path']}")
    mel = preprocess(wave).to(device)
    with torch.no_grad():
        target = torch.cat(
            [
                model.style_encoder(mel.unsqueeze(1)),
                model.predictor_encoder(mel.unsqueeze(1)),
            ],
            dim=-1,
        )
        token_tensor = tokens.unsqueeze(0).to(device)
        lengths = torch.tensor([token_tensor.shape[1]], device=device)
        text_mask = length_to_mask(lengths).to(device)
        bert = model.bert(token_tensor, attention_mask=(~text_mask).int())
    return target, bert


def _reference_style(
    model,
    record: Dict[str, Any],
    cache: Dict[str, torch.Tensor],
    device: str,
) -> torch.Tensor:
    key = str(record["ref_clip_id"])
    if key not in cache:
        cache[key] = extract_style(model, record["ref_wav_path"], device)
    return cache[key]


def train_one_epoch(
    model,
    records: List[Dict[str, Any]],
    tokenizer: Tuple[Any, Any],
    reference_cache: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    args: argparse.Namespace,
    epoch: int,
    state: Dict[str, Any],
) -> Tuple[float, int, bool]:
    model.diffusion.unet.train()
    order = list(records)
    random.Random(args.seed + epoch).shuffle(order)
    total_loss = 0.0
    seen = 0
    pending = 0
    optimizer.zero_grad(set_to_none=True)
    for record in order:
        clip_id = str(record["clip_id"])
        if clip_id in state["failed"]:
            continue
        try:
            target, bert = prepare_example(model, record, tokenizer, args.device)
            reference = _reference_style(
                model, record, reference_cache, args.device
            )
            loss = model.diffusion(
                target.unsqueeze(1),
                embedding=bert,
                features=reference,
            )
        except Exception as exc:  # per-clip guard, reported not hidden
            state["failed"].add(clip_id)
            state["failures"].append(
                {"clip_id": clip_id, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue

        (loss / max(1, args.grad_accum)).backward()
        total_loss += float(loss.detach())
        seen += 1
        pending += 1
        if pending >= max(1, args.grad_accum):
            torch.nn.utils.clip_grad_norm_(state["trainable"], args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            state["global_step"] += 1
            if state["global_step"] % max(1, args.log_every) == 0:
                print(
                    f"[train] epoch {epoch} step {state['global_step']} "
                    f"loss {float(loss.detach()):.4f}",
                    flush=True,
                )
            if args.max_steps and state["global_step"] >= args.max_steps:
                return total_loss / max(1, seen), seen, True

    if pending:
        torch.nn.utils.clip_grad_norm_(state["trainable"], args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        state["global_step"] += 1
    return total_loss / max(1, seen), seen, False


def evaluate_loss(
    model,
    records: List[Dict[str, Any]],
    tokenizer: Tuple[Any, Any],
    reference_cache: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    state: Dict[str, Any],
) -> Tuple[Optional[float], int]:
    model.diffusion.unet.eval()
    total = 0.0
    seen = 0
    with torch.no_grad():
        for record in records:
            clip_id = str(record["clip_id"])
            if clip_id in state["failed"]:
                continue
            try:
                target, bert = prepare_example(model, record, tokenizer, args.device)
                reference = _reference_style(
                    model, record, reference_cache, args.device
                )
                losses = [
                    float(
                        model.diffusion(
                            target.unsqueeze(1),
                            embedding=bert,
                            features=reference,
                            embedding_mask_proba=0.0,
                        ).detach()
                    )
                    for _ in range(max(1, args.val_repeats))
                ]
            except Exception as exc:
                state["failed"].add(clip_id)
                state["failures"].append(
                    {"clip_id": clip_id, "error": f"{type(exc).__name__}: {exc}"}
                )
                continue
            total += float(np.mean(losses))
            seen += 1
    return (total / seen if seen else None), seen


def _safe_name(clip_id: str) -> str:
    return str(clip_id).replace("/", "__").replace("\\", "__")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def evaluate_generation(
    synthesizer,
    model,
    records: List[Dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Generate paired base/LoRA clips and measure real objective metrics."""
    import soundfile as sf

    from voix.models.tts_backend import _lufs_normalize

    # Programmatic callers built namespaces before this flag existed; treat a
    # missing attribute as the documented default rather than crashing.
    generate_only = bool(getattr(args, "eval_generate_only", False))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.jsonl").write_text("", encoding="utf-8")
    reference_cache: Dict[str, torch.Tensor] = {}
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for index, record in enumerate(records):
        reference = _reference_style(
            model, record, reference_cache, args.device
        )
        for condition in ("base", "lora"):
            set_lora_enabled(model.diffusion.unet, condition == "lora")
            # Same noise for both conditions: the only difference is the adapter.
            seed = int(args.seed) + 1000 * index
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                wave = synthesizer.inference(
                    text=str(record["text"]),
                    ref_s=reference,
                    alpha=args.alpha,
                    beta=args.beta,
                    diffusion_steps=args.eval_diffusion_steps,
                )
            normalized = _lufs_normalize(
                torch.from_numpy(np.asarray(wave, dtype=np.float32)),
                24000,
                target_lufs=-23.0,
            )
            path = output_dir / f"{_safe_name(record['clip_id'])}__{condition}.wav"
            sf.write(str(path), normalized.squeeze(0).numpy(), 24000)
            if generate_only:
                row = {
                    "clip_id": str(record["clip_id"]),
                    "speaker_id": str(record["speaker_id"]),
                    "condition": condition,
                    "text": str(record["text"]),
                    "wav_path": str(path),
                    "target_wav_path": str(record["wav_path"]),
                    "ref_wav_path": str(record["ref_wav_path"]),
                    "wer": None,
                }
            else:
                from voix.evaluation.metrics import (
                    prosody_statistics,
                    speaker_similarity,
                )

                stats = prosody_statistics(path, sample_rate=24000)
                row = {
                    "clip_id": str(record["clip_id"]),
                    "speaker_id": str(record["speaker_id"]),
                    "condition": condition,
                    "text": str(record["text"]),
                    "wav_path": str(path),
                    "wer": None,
                    "speaker_similarity_target": speaker_similarity(
                        path, record["wav_path"], device=args.eval_device
                    ),
                    "speaker_similarity_ref": speaker_similarity(
                        path, record["ref_wav_path"], device=args.eval_device
                    ),
                    **stats,
                }
            rows.append(row)
            # A 4 GB card cannot hold the full model plus a long-lived CUDA
            # cache across 200+ generations.  Release it after every clip and
            # persist the row so a native crash cannot lose completed work.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with (output_dir / "rows.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rows[-1]) + "\n")
            print(
                f"[eval] {index + 1}/{len(records)} {condition} "
                f"{record['clip_id']}",
                flush=True,
            )
    set_lora_enabled(model.diffusion.unet, True)

    wer_status: Dict[str, Any] = {}
    if generate_only:
        wer_status = {
            "status": "skipped",
            "reason": "--eval-generate-only; run scripts/evaluation/evaluate_n1_metrics.py",
        }
    else:
        from voix.evaluation.audio_metrics import transcribe_wer

        for condition in ("base", "lora"):
            subset = [row for row in rows if row["condition"] == condition]
            result = transcribe_wer(
                [row["wav_path"] for row in subset],
                [row["text"] for row in subset],
                model_size=args.whisper_model,
                device=args.eval_device,
            )
            wer_status[condition] = {
                key: value
                for key, value in result.items()
                if key not in {"per_sample_wer", "transcripts"}
            }
            if result.get("status") == "computed":
                for row, wer in zip(subset, result["per_sample_wer"]):
                    row["wer"] = float(wer)

    summary = summarize_generation(rows)
    summary["wer_status"] = wer_status
    summary["elapsed_sec"] = round(time.time() - started, 1)
    return {"status": "computed", "summary": summary, "rows": rows}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument("--eval-output", type=Path, default=None)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--load-on-cpu",
        action="store_true",
        help=(
            "Construct StyleTTS2 on CPU, then move only the training modules "
            "(BERT, style/predictor encoders, diffusion U-Net) to --device. "
            "This is a low-VRAM training mode; it is incompatible with "
            "--eval-samples > 0 because the decoder/vocoder are released."
        ),
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--val-repeats", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--eval-generate-only",
        action="store_true",
        help=(
            "Generate paired wavs without ECAPA/prosody/WER metrics.  Use on "
            "low-RAM machines, then run scripts/evaluation/evaluate_n1_metrics.py."
        ),
    )
    parser.add_argument("--eval-samples", type=int, default=25)
    parser.add_argument("--eval-diffusion-steps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument("--beta", type=float, default=0.22)
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--eval-device", default="cpu")
    return parser.parse_args(argv)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _construct_synthesizer(load_on_cpu: bool):
    """Load StyleTTS2, optionally forcing the initial placement to CPU.

    The 4 GB GPU cannot hold the full StyleTTS2 stack (text aligner, F0
    extractor, BERT, encoders, diffusion, decoder, vocoder) at once.  The
    training loop needs only BERT, the style/predictor encoders, and the
    diffusion U-Net.  Forcing the constructor to CPU lets us move that subset
    to the GPU and release the inference-only modules.
    """
    from styletts2 import tts

    if not load_on_cpu:
        return tts.StyleTTS2()
    original = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        return tts.StyleTTS2()
    finally:
        torch.cuda.is_available = original


def _load_runtime(args: argparse.Namespace):
    from styletts2.text_utils import TextCleaner

    print("[setup] loading StyleTTS2", flush=True)
    started = time.time()
    synthesizer = _construct_synthesizer(bool(getattr(args, "load_on_cpu", False)))
    model = synthesizer.model
    if getattr(args, "load_on_cpu", False):
        if args.eval_samples > 0:
            raise ValueError(
                "--load-on-cpu releases the decoder/vocoder and cannot run "
                "generation evaluation; set --eval-samples 0"
            )
        training_modules = ("bert", "style_encoder", "predictor_encoder", "diffusion")
        for key in training_modules:
            model[key].to(args.device)
        for key in ("text_aligner", "pitch_extractor", "decoder", "vocoder"):
            if key in model:
                del model[key]
        synthesizer.device = str(args.device)
    elif str(synthesizer.device) != str(args.device):
        for key in model:
            model[key].to(args.device)
        synthesizer.device = str(args.device)
    print(f"[setup] loaded in {time.time() - started:.1f}s", flush=True)

    for key in model:
        for parameter in model[key].parameters():
            parameter.requires_grad = False
        model[key].eval()
    targets = inject_lora(
        model.diffusion.unet,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    trainable = [
        parameter
        for parameter in model.diffusion.unet.parameters()
        if parameter.requires_grad
    ]
    print(
        f"[setup] injected LoRA into {len(targets)} layers, "
        f"{count_trainable_parameters(model.diffusion.unet):,} trainable params",
        flush=True,
    )
    tokenizer = (synthesizer.phoneme_converter, TextCleaner())
    return synthesizer, model, targets, trainable, tokenizer


def run(args: argparse.Namespace) -> Dict[str, Any]:
    _seed_everything(args.seed)
    frame = load_manifest(args.manifest)
    train_records = frame[frame["split"] == "train"].to_dict("records")
    val_records = frame[frame["split"] == "val"].to_dict("records")
    print(
        f"[data] {len(train_records)} train clips / "
        f"{frame.loc[frame.split == 'train', 'speaker_id'].nunique()} speakers; "
        f"{len(val_records)} val clips / "
        f"{frame.loc[frame.split == 'val', 'speaker_id'].nunique()} speakers",
        flush=True,
    )

    synthesizer, model, targets, trainable, tokenizer = _load_runtime(args)
    reference_cache: Dict[str, torch.Tensor] = {}

    if args.eval_only:
        payload = load_lora(model.diffusion.unet, args.output)
        print(
            f"[eval] loaded {args.output} "
            f"(epoch {payload.get('metadata', {}).get('epoch')})",
            flush=True,
        )
        if args.eval_samples <= 0:
            raise ValueError("--eval-only requires --eval-samples > 0")
        records = select_eval_records(frame, args.eval_samples, args.seed)
        result = evaluate_generation(
            synthesizer, model, records, args.eval_dir, args
        )
        eval_output = args.eval_output or (
            args.output.with_name(args.output.stem + "_generation_eval.json")
        )
        eval_output.parent.mkdir(parents=True, exist_ok=True)
        eval_output.write_text(
            json.dumps(_jsonable(result), indent=2), encoding="utf-8"
        )
        print(json.dumps(result["summary"], indent=2), flush=True)
        print(f"Wrote {eval_output}", flush=True)
        return result

    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = max(
        1, math.ceil(len(train_records) / max(1, args.grad_accum))
    )
    total_steps = steps_per_epoch * max(1, args.epochs)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * max(0.0, args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_cosine(
            step, total_steps, warmup_steps, args.min_lr_ratio
        ),
    )

    state: Dict[str, Any] = {
        "trainable": trainable,
        "global_step": 0,
        "failed": set(),
        "failures": [],
    }
    history: List[Dict[str, Any]] = []
    best_val: Optional[float] = None
    best_epoch = -1
    started = time.time()
    history_path = args.history or args.output.with_name(
        args.output.stem + ".history.json"
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max(1, args.epochs) + 1):
        train_loss, train_n, hit_limit = train_one_epoch(
            model,
            train_records,
            tokenizer,
            reference_cache,
            optimizer,
            scheduler,
            args,
            epoch,
            state,
        )
        val_loss, val_n = evaluate_loss(
            model, val_records, tokenizer, reference_cache, args, state
        )
        row = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 6),
            "train_clips": train_n,
            "val_loss": round(float(val_loss), 6) if val_loss is not None else None,
            "val_clips": val_n,
            "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
            "global_step": state["global_step"],
        }
        history.append(row)
        print(
            f"[epoch {epoch}] train {row['train_loss']:.4f} "
            f"({train_n} clips) val {row['val_loss']} ({val_n} clips) "
            f"lr {row['lr']:.2e}",
            flush=True,
        )

        if val_loss is not None and (best_val is None or val_loss < best_val - 1e-6):
            best_val = float(val_loss)
            best_epoch = epoch
            metadata = {
                "schema_version": "voix.styletts2_lora.v1",
                "base_model": "StyleTTS2-LibriTTS (yl4579/StyleTTS2-LibriTTS)",
                "manifest": str(args.manifest),
                "manifest_sha256": manifest_digest(frame),
                "train": {
                    "clips": len(train_records),
                    "speakers": int(
                        frame.loc[frame.split == "train", "speaker_id"].nunique()
                    ),
                    "hours": round(
                        float(
                            frame.loc[frame.split == "train", "duration_sec"].sum()
                        )
                        / 3600.0,
                        3,
                    ),
                },
                "val": {
                    "clips": len(val_records),
                    "speakers": int(
                        frame.loc[frame.split == "val", "speaker_id"].nunique()
                    ),
                    "hours": round(
                        float(frame.loc[frame.split == "val", "duration_sec"].sum())
                        / 3600.0,
                        3,
                    ),
                },
                "injected_layers": targets,
                "trainable_parameters": count_trainable_parameters(
                    model.diffusion.unet
                ),
                "adapter": adapter_statistics(model.diffusion.unet),
                "epoch": epoch,
                "train_loss": row["train_loss"],
                "val_loss": row["val_loss"],
                "history": history,
                "failures": state["failures"],
                "args": _jsonable(vars(args)),
                "notes": [
                    "Research pilot: only the diffusion U-Net is adapted with LoRA.",
                    "The identity path (CFVN/CVAE/adapter) is frozen and untouched.",
                    "VoxCeleb2 transcripts are Whisper ASR, not ground-truth text.",
                    "No MOS or naturalness claim is made from these objective metrics.",
                ],
            }
            save_lora(model.diffusion.unet, args.output, metadata=metadata)
            print(f"[checkpoint] wrote {args.output}", flush=True)

        history_path.write_text(
            json.dumps(_jsonable({"history": history, "best_epoch": best_epoch}),
                       indent=2),
            encoding="utf-8",
        )
        if hit_limit:
            print("[train] reached --max-steps", flush=True)
            break
        if best_epoch > 0 and epoch - best_epoch >= max(1, args.patience):
            print(f"[train] early stop after epoch {epoch}", flush=True)
            break

    if best_epoch < 0:
        raise RuntimeError("no validation loss was computed; nothing to save")

    last_path = args.output.with_name(args.output.stem + "_last.pt")
    save_lora(
        model.diffusion.unet,
        last_path,
        metadata={
            "epoch": history[-1]["epoch"],
            "val_loss": history[-1]["val_loss"],
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "history": history,
            "failures": state["failures"],
            "args": _jsonable(vars(args)),
        },
    )
    print(f"[checkpoint] wrote {last_path}", flush=True)

    payload = load_lora(model.diffusion.unet, args.output)
    adapter = adapter_statistics(model.diffusion.unet)
    print(f"[checkpoint] adapter statistics: {adapter}", flush=True)
    result: Dict[str, Any] = {
        "status": "computed",
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "epochs_run": len(history),
        "elapsed_sec": round(time.time() - started, 1),
        "history": history,
        "failures": state["failures"],
        "adapter": adapter,
        "checkpoint": str(args.output),
    }

    if args.eval_samples > 0:
        print(
            f"[eval] loaded best checkpoint epoch "
            f"{payload.get('metadata', {}).get('epoch')}",
            flush=True,
        )
        records = select_eval_records(frame, args.eval_samples, args.seed)
        generation = evaluate_generation(
            synthesizer, model, records, args.eval_dir, args
        )
        result["generation"] = generation["summary"]
        eval_output = args.eval_output or (
            args.output.with_name(args.output.stem + "_generation_eval.json")
        )
        eval_output.parent.mkdir(parents=True, exist_ok=True)
        eval_output.write_text(
            json.dumps(_jsonable(generation), indent=2), encoding="utf-8"
        )
        print(f"[eval] wrote {eval_output}", flush=True)
    else:
        result["generation"] = {
            "status": "unavailable",
            "reason": "--eval-samples 0",
        }

    print(json.dumps(_jsonable(result), indent=2), flush=True)
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
