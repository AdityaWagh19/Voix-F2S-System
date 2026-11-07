# VOIX repository cleanup inventory

**Status: executed on 2026-11-05 after explicit user approval.** This document
preserves the pre-execution inventory; the authoritative file-level record is
`results/audit/cleanup/`. The cleanup archived 7,053 files (936.7 MiB) to
`D:\voix_archive\20261105_cleanup` and deleted 170 reproducible cache files
(159.8 MiB). Every archived file was re-hashed at its destination. No canonical
checkpoint, v1/v2 demo audio, manifest-pinned audit artifact, or user work was
touched. The inventory was measured from `D:\voix` (branch `main` at `3bbf21d`).

The working tree contained roughly 400 status entries: 59 modified tracked
files, 33 deleted tracked files, and 308 untracked files. Those are treated as
user work and are **not** cleanup candidates. The exact count moves as audit
artifacts are written; `results/audit/final_manifest.json` records the frozen
git state at the end of the audit. In particular, the modified
`demo/` showcase assets and the deleted legacy demo/benchmark scripts must not
be reverted by this cleanup.

Sizes are directory totals on disk and are rounded. They include generated
audio, JSON, logs, and checkpoints. "Archive" means move to an approved
external/attic location after preserving manifests and hashes; it does not mean
delete.

## Execution record

- Archive: `D:\voix_archive\20261105_cleanup\files\...`; 7,053 files, 936.7 MiB.
  Includes 815.9 MiB of intermediate audio, 113.6 MiB of experimental
  checkpoints, and the 7.2 MiB `results/demo_pre_phase0/` snapshot.
- Deletion: 170 reproducible cache files, 159.8 MiB, from `.tmp/`,
  `results/audio_quality*/`, `results/audio_samples/`, `results/control_audit*/`,
  `results/demo_face_audit/`, and `results/step_ablation/`.
- Per-file SHA-256 records: `results/audit/cleanup/archive_manifest.json` and
  `deletion_manifest.json`; execution summary: `results/audit/cleanup/README.md`.
- Retained: the canonical cfvn/cvae/style-adapter trio, `cfvn_v2_arcface.pt`,
  `D_joint_infonce_best.pt`, the N1 pilot checkpoints, the StyleTTS2 LoRA pilot
  checkpoint, `results/styletts2_lora_eval/`, all v1/v2 demo audio, and every
  final-manifest evidence file.
- Consequence: historical experiment JSON files and some docs remain, but they
  may reference archived `.pt` or WAV paths. This is intentional; the summaries
  and hashes were kept for provenance and the archived files are listed by
  source path.

## Decision summary

| Class | Approx. size | Recommendation |
|---|---:|---|
| Canonical checkpoints, pretrained models, datasets | ~17.2 GB | **Retain**. Do not delete. |
| Final v2 demo and audit evidence | ~96 MB | **Retain**. These are the deliverables. |
| Reproducible diagnostic/cache output | ~180 MB | **Deleted after approval** (159.8 MiB). |
| Evidence-bearing intermediate audio | ~800 MB | **Archived after approval** (815.9 MiB); JSON/MD summaries kept. |
| Experimental checkpoints and smoke artifacts | ~130 MB | **Archived after approval** (113.6 MiB); canonical trio and eval JSON kept. |
| Modified/deleted/untracked user work | unknown | **Do not touch**. |

The largest single reclaim opportunity is the `.tmp` cache (84.9 MB), followed
by legacy benchmark audio (39.9 MB) and old audio-quality diagnostic audio
(25.4 MB). None of these are required by the v2 demo or by the audit reports.

## Inventory

### 1. Retain: canonical and serving-critical

| Path | Size | Contents / why it exists | Proposed action |
|---|---:|---|---|
| `results/checkpoints/cfvn_v2_best.pt` | 3.0 MB | Canonical CFVN checkpoint, SHA-256 `0253bbbf…` | Retain. Hash-pinned in `results/canonical_hashes.json` and in the demo generator. |
| `checkpoints/cvae_cfvn_face192_latent128_diversity.pt` | 5.5 MB | Canonical CVAE checkpoint, SHA-256 `bb7ad29b…` | Retain. |
| `checkpoints/style_adapter_cfvn_best.pt` | 0.7 MB | Canonical StyleAdapter checkpoint, SHA-256 `97750ee8…` | Retain. |
| `checkpoints/models/` | 325.5 MB | StyleTTS2 and other pretrained model files | Retain. Required for reproduction/serving. |
| `checkpoints/speechbrain/` | 84.9 MB | ECAPA speaker-embedding model | Retain. Used by every demo/audit verification. |
| `checkpoints/mediapipe/` | 3.6 MB | Face-mesh model | Retain. |
| `checkpoints/emotion_adapter/` | 2.3 MB | E2 emotion adapter experiments | Retain; referenced by `results/emotion/FINDINGS.md`. |
| `checkpoints/emotion_residuals/`, `checkpoints/accent_residuals/` | <1 MB | E1/A3 class-mean artifacts | Retain; hashed in the v2 report. |
| `data/raw/` | 12.35 GB | Downloaded source corpora | Retain. Re-downloading is expensive and split reproduction depends on it. |
| `data/processed/` | 4.42 GB | Processed manifests, features, and splits | Retain. Required for retraining and audit reproduction. |
| `results/demo/research_v2/` | see report | Final 21-clip demo (audio, faces, report, CSV, README, HTML) | Retain. This is the current deliverable. |
| `results/audit/` | 82.9 MB | Baseline, calibration, diversity, condition-response, and before/after evidence | Retain. This is the audit record. |
| `results/emotion/` | 601.5 MB | E1/E2 evidence, generated audio, findings | Retain. Archive only intermediate audio after approval, never the findings/JSON. |
| `results/naturalness/` | 189.5 MB | N1 LoRA pilot, gate report, evaluation audio | Retain the JSON/MD; audio is an archive candidate. |
| `results/accent/` | 51.9 MB | A0/A2/A3 evidence and generated audio | Retain the JSON/MD; audio is an archive candidate. |
| `results/demo/research/` | part of `results/demo` | v1 gallery, preserved for before/after | **Retained**; the user chose to keep both v1 and v2. |
| `results/demo_pre_phase0/` | 7.2 MB | Pre-phase-0 demo snapshot referenced by the top-level README and AUDIT_REPORT | **Archived** on 2026-11-05; both documents now point to the archive path. |

### 2. Delete-after-approval: reproducible caches and legacy diagnostics

| Path | Size | Files | Contents / references | Risk if removed |
|---|---:|---:|---|---|
| `.tmp/` | 84.9 MB | 12 | `voix_ffmpeg` (83.6 MB), `formant_probe`, matplotlib cache, pytest temp dirs, one scratch test | Low. Caches and a downloaded ffmpeg binary are re-creatable. Confirm no run is using them first. |
| `results/audio_quality/` | 12.6 MB | 30 | Old `diagnose_audio_quality.py` audio | Low. Summary JSON lives in `results/audio_quality_diagnostics.json`. |
| `results/audio_quality_20261029/` | 7.7 MB | 18 | Older copy of the same diagnostic | Low. Superseded. |
| `results/audio_quality_phase0/` | 5.1 MB | 12 | Phase-0 diagnostic audio | Low. Superseded. |
| `results/audio_samples/` | 39.9 MB | 60 | Legacy retrieval-benchmark audio (`D_joint_infonce_opt`, etc.) | Low–medium. Only referenced by `legacy/scripts/benchmark*.py`, not by the current pipeline. |
| `results/control_audit/` | 3.5 MB | 8 | Older control-condition audit audio and JSON | Low–medium. Superseded by `results/audit/condition_response/`; confirm no external citation. |
| `results/control_audit_t1/` | 1.7 MB | 6 | Earlier control audit | Low–medium. Superseded. |
| `results/demo_face_audit/` | 3.9 MB | 11 | Older face-condition audit | Low–medium. Superseded by the face-diversity and demo-settings audits. |
| `results/step_ablation/` | 0.5 MB | 7 | Step-sweep audio | Low. Summary is in `results/step_ablation.json`. |

Subtotal: roughly **159 MB** if all are deleted. This is the safest reclaim
group, but it still requires approval because some files are cited as historical
evidence in the top-level audit.

### 3. Archive-after-approval: evidence-bearing intermediate audio

These are not junk. They support published/recorded negative or exploratory
results. Archive them to external storage with a checksum manifest, or retain
them if disk space is not a constraint.

| Path | Size | Contents | Proposed action |
|---|---:|---|---|
| `results/emotion/e0/`, `e0_smoke/`, `e0_smoke_v2/` | ~43 MB | Emotion pilot generation/evaluation audio | Archive if the final E1/E2 JSON and findings are sufficient. |
| `results/emotion/e1/`, `e1c/`, `e1_smoke/` | ~170 MB | E1 class-mean and corrected-control audio | Archive only after the E1 report is frozen. |
| `results/emotion/e2/`, `e2c/`, `e2_smoke/` | ~387 MB | E2 learned-adapter and corrected-control audio | Archive only after the E2 negative result is frozen. |
| `results/naturalness/n0b/` | 110.5 MB | Pre-LoRA naturalness evaluation audio | Archive; keep JSON/MD. |
| `results/naturalness/n1/` | 67.7 MB | N1 LoRA paired generation/evaluation audio | Archive; keep `gate_report.json`, `EVAL_METHOD.md`, and the LoRA checkpoint. |
| `results/accent/a2/`, `a2b/`, `a3/` | ~52 MB | Accent residual evaluation audio | Archive; keep `FINDINGS.md`, `scale_selection.json`, and the classifier. |
| `results/styletts2_lora_eval/` | 17.8 MB | 52 base/LoRA evaluation WAVs referenced by two JSON files (54 path references) | **Retained**; archive only if the two generation-eval JSON files are rewritten first. |
| `results/demo_pre_phase0/` | 7.2 MB | Previous demo snapshot | **Archived** on 2026-11-05; README/AUDIT_REPORT now point to the archive. |
| `results/demo/research/` (v1) | ~4 MB | 12-clip v1 gallery | **Retained**; the user chose to keep both v1 and v2. |

Subtotal: roughly **860 MB** of audio. Most of the disk total in `results` is
this audio, not the checkpoints.

### 4. Archive-after-approval: experimental checkpoints and smoke artifacts

Keep the canonical trio and the N1 LoRA checkpoint used for the negative result.
The rest are experiment history and can be archived.

| Path | Approx. size | Examples | Proposed action |
|---|---:|---|---|
| `results/checkpoints/` non-canonical files | ~87 MB | `cfvn_struct_*.pt`, `cfvn_smoke*.pt`, `cfvn_stability.pt`, `cfvn_v2_arcface.pt`, `A/B/C/D_*`, `cvae_smoke*`, eval JSON | Archive `.pt` files; keep eval JSON if it records a reported result. **Never remove `cfvn_v2_best.pt`.** |
| `checkpoints/` non-canonical top-level files | ~41 MB | `cvae_smoke.pt`, `cvae_v2_spkavg_best.pt`, `cvae_v3_*.pt`, `style_adapter_cfvn_5clip_*`, `style_adapter_best.pt`, `adapter_smoke_pilot100.pt` | Archive `.pt`/history files after confirming no script default points to them. **Never remove the canonical CVAE or StyleAdapter.** |
| `results/naturalness/n1/ami_lora_smoke*.pt` | ~3.5 MB | Smoke LoRA weights | Archive; keep `ami_lora_pilot.pt`. |
| `results/checkpoints/styletts2_lora_smoke*`, `styletts2_lora_prosody_last.pt` | ~5 MB | LoRA training checkpoints | Archive; keep `ami_lora_pilot.pt` and the eval JSON. |
| `checkpoints/*.history.json`, `*.eval.json` | <1 MB | Training history/evaluation metadata | Retain; cheap and useful provenance. |

### 5. Do not touch

- The 59 modified tracked files, including `README.md`, `configs/*.yaml`,
  `demo/*` audio and metadata, `pyproject.toml`, and the top-level
  `AUDIT_REPORT.md`.
- The 33 deleted tracked files. They may be intentional user changes; cleanup
  must not restore them.
- The 308 untracked files outside `results/audit/`, `results/demo/research_v2/`,
  and the new audit scripts. Many are user work or generated evidence.
- `results/candidate_pool6/` (2.6 MB). This contains the six candidate voices
  and metadata used in the voice-selection decision; keep it until the selection
  report is frozen.
- Any `.wav` referenced by `report.json` in `results/demo/research_v2/` or by a
  `results/audit/*` report.

## Suggested execution order (only after approval)

1. **Freeze evidence first.** Re-run `python scripts/build_final_manifest.py`
   and the relevant audit tests, then record the current hashes.
2. **Create an archive manifest** (path, size, SHA-256, source report) for every
   file proposed for archive.
3. **Delete caches/legacy diagnostics only** from group 2. Do not touch group 1
   or group 5.
4. **Archive group 3 and 4** to an external drive or `D:\voix_archive\`, then
   update any JSON path references before removing the originals.
5. **Promote the v2 demo** only after the user approves replacing
   `results/demo/research/`. Until then, keep both galleries.
6. **Update docs** (`README.md` §6.2 and top-level `AUDIT_REPORT.md` §10.9) to
   point at the new audit evidence and to remove the stale claim that the
   residual is not a face-style/vocoder hiss issue.

## What this inventory does not do

- It does not delete, move, or rename anything.
- It does not modify canonical checkpoints.
- It does not claim that archived audio is worthless; it separates
  reproducible diagnostics from evidence-bearing negative results.
- It does not clean the user's uncommitted work.
