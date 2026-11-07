# VOIX Scope Decision

**Status:** Active — research/demo only  
**Date:** 2026-10-28  
**Decision owner:** Project maintainers

This document is the scope of record for the next phase of VOIX. It supersedes
older demo and MVP claims where they conflict. It describes a **research/demo
system**, not a production or commercial deployment. No result in this document
is a production-readiness claim.

> **2026-11-05 update:** the current research demo is the v2 21-clip surface at
> `results/demo/research_v2/`: three voices, three emotions (happy, sad, angry),
> and three accents (US, England, Australian). The 15-asset measurements in §4
> are the v1 baseline and are retained for comparison. Audited v2 results and
> non-claims are in `results/audit/AUDIT_REPORT.md`.

## 1. Product thesis

VOIX maps a face image to a **distribution of plausible speaker identities**,
not a single deterministic voice. The core scientific and product claim is
face-conditioned voice identity plus controllable one-to-many generation.

Accent and emotion are **user-selected expression controls** on top of that
identity. They are not inferred from the face, and they are not the project's
core differentiator. They matter because a usable voice product must be able to
sound expressive and regionally appropriate.

## 2. In scope

| Area | Role | Current status |
|---|---|---|
| Candidates / one-to-many face-to-voice generation | Core | Working end-to-end; identity fidelity still needs improvement |
| Emotions | First expression control (research/demo) | Wired and measurable; heuristic, not learned or perceptually validated; measured gate fails |
| Accents | Second expression control (research/demo) | Wired and measurable; heuristic and can damage intelligibility; measured gate fails |

## 3. Out of scope permanently

The following are removed from the product surface, public claims, demo
showcase, roadmap, and new engineering work:

- Cross-lingual synthesis (Spanish, French, German, Italian).
- Speech-difference / clinical speech simulation.
- Craniofacial morphology sliders as a user-facing control.

Existing code, checkpoints, tests, and generated assets may remain in the
repository for reproducibility and historical reference. They must not be
advertised as supported product capabilities or used as evidence for the
in-scope claims.

## 4. Why this scope

Measured evidence from the v1 15-asset in-scope demo and controlled audits:

- Candidate generation is the cleanest category. On the v1 15-asset demo
  the three candidates have pairwise ECAPA similarity 0.7236 mean / 0.5408 min,
  so they no longer collapse into one identity (the old demo's 1-vs-3 pair was
  0.9393). Face→voice cosine is still weak: 0.1019 mean for candidates and
  0.0857 mean for expression variants.
- Emotion controls move F0 and duration in the intended direction for happy,
  angry, and surprised, and the sad lowering is now present (211.0 Hz versus
  neutral 234.4 Hz). Whispered is now quieter than neutral (energy 0.0398
  versus 0.0513), which became measurable after the demo writer stopped
  peak-normalizing. The controls are still not production-grade: on the v1
  12 identity-controlled expression assets the speaker-preservation mean is
  0.509 (min 0.236, sad) against 0.80, and the worst emotion WER is 0.235
  (angry). These are single-clip diagnostics.
- Accent is weaker. The v1 accent WER mean is 0.196, with Indian English
  at 0.353; the worst overall WER is 0.353 against 0.20. The v1 path was
  regex respelling plus fixed style deltas, not phonological accent control.
- Cross-lingual is not real multilingual TTS: the archived 29-asset run had mean
  WER 1.246 and the path was English G2P/respelling. It is out of scope.
- Craniofacial sliders were found to be no-ops or actively wrong in the audit.
- Speech-difference simulation is clinically sensitive and has no clinical or
  perceptual validation.

The project also has a weak core identity link that must be improved before the
system can support production claims:

- Face–voice cosine on the v1 15-asset demo: 0.1019 mean for candidates and
  0.0857 mean for expression variants.
- Expression identity preservation against the primary candidate: 0.5798 mean /
  0.2577 min (sad), below the 0.80 gate.
- CFVN official-test face→voice R@1: 0.0932 (11.0× chance), R@10: 0.4068.
- CFVN face-face R@1 is 0.8475 vs raw 0.9153. A structure-preservation
  distillation term raised test face-face R@1 to 0.8814–0.9153, but only 1 of 3
  seeds matched raw, so it is not adopted (see `AUDIT_REPORT.md`).

## 5. Balanced training data and control inventory

Any future production control must be trained on **class-balanced,
speaker-disjoint, appropriately licensed** data. For the current research/demo
phase, VoxCeleb2 remains research/non-commercial only, and no commercial-use
claim is made. “Balanced” is an engineering requirement, not a marketing claim:

- comparable speakers and hours per class after sampling (target max/min ≤ 1.5);
- at least 30 speakers and 2 hours per trained class;
- a per-speaker cap (for example 30 clips) so a few speakers cannot dominate;
- speaker-disjoint train/validation/test splits;
- quality and transcript filters applied before balancing;
- no class exposed in the product until it passes the expression gates.

The exact inventory must be selected from measured speaker counts, not from
marketing appeal. The proposed inventory below is the target; it is not yet a
claim that every class is trainable.

### 5.1 Emotion inventory

Current demo: neutral, happy, sad, angry, whispered, surprised.

Assessment: this is not the standard basic-emotion set. Standard expressive-TTS
inventories use anger, disgust, fear, happiness, sadness, and surprise, usually
with neutral added. Whispered is a phonation/voice-quality style, not an
emotion.

Recommended training inventory: **neutral, happy, sad, angry, fear, disgust**
(the six CREMA-D classes). Add **surprised** only when a commercially usable
surprise corpus with enough speakers exists (RAVDESS commercial license or
MSP-Podcast agreement). Move **whispered** to a separate voice-quality control,
or remove it; do not evaluate it as an emotion.

### 5.2 Accent inventory

Current demo: General American, British RP, Australian, Indian English, Irish,
Scottish.

Assessment: these are recognizable and market-relevant, but not balanced.
“British RP” is a specific prestige variety while the available corpus label is
“England English”; Scottish is thin in GLOBE_V3; Canadian is missing despite
strong coverage.

Recommended training inventory, subject to speaker-count verification:
**US English (General American), British English (England English; do not call
it RP without RP-specific data), Australian English, Canadian English,
Indian/South Asian English, and Irish English.**

Evaluation-only until they pass the 30-speaker / 2-hour threshold: Scottish,
Welsh, New Zealand, South African, and Northern Irish. If the product requires
Scottish, supplement it with OpenSLR 83 (39 speakers) and treat it as a small,
limited class rather than a balanced one.

Primary data sources: CREMA-D for pilot/evaluation emotion work; MSP-Podcast
after the institutional agreement for learned emotion control and naturalistic
generalization; GLOBE_V3 for accent. CREMA-D is about 0.88 h per emotion, below
the 2 h/class gate, so it cannot train a learned control. GLOBE_V3 does
not expose `speaker_id`; join `common_voice_path` to Common Voice 21 metadata to
recover `client_id` before constructing speaker-disjoint splits.

### 5.3 Access, consent, and disk requirements

These are the concrete actions required before training, not assumptions. The
same information is in [`docs/MASTER.md`](docs/MASTER.md) §3.

- **CREMA-D (emotion, pilot/evaluation).** Direct download from the
  `CheyneyComputerScience/CREMA-D` GitHub repository. License: ODbL 1.0; no
  access form. Attribute and cite the dataset. Size: 7,442 clips, 91 actors,
  5.26 h, six emotion classes (~0.88 h/class). It fails the 2 h/class gate, so
  it is evaluation-only and cannot train a learned emotion control.
- **MSP-Podcast (emotion, naturalistic).** Academic license, free of cost but
  requiring an institutional signatory. Download the license PDF from the
  MSP lab site, have the institution sign it, and email Prof. Carlos Busso.
  Size: 264,705 turns, 409 h, 2,220 train speakers. Do not download or train
  before the agreement is executed.
- **GLOBE_V3 (accent, primary).** Hugging Face `MushanW/GLOBE_V3`, CC0 1.0.
  The repository tree is **348.7 GB across 603 files**: 165.5 GB of
  `data/*.parquet` plus 183.2 GB of duplicate `train/`/`dev/`/`test/` audio.
  The card's 165.5 GB download size is the Parquet payload; 184.5 GB is the
  uncompressed dataset size. The `D:` drive has only **138.1 GB free**, so a
  full clone does not fit and is unnecessary. The metadata pass reads each
  Parquet footer plus the ~20 KB metadata span per row group (about 150 MB for
  all 187 shards), and the audio pass fetches only selected row groups.
- **Common Voice 21 metadata (required join).** GLOBE_V3 has no `speaker_id`.
  Download the Common Voice 21 English metadata separately and join
  `common_voice_path` to `client_id` before constructing speaker-disjoint
  splits. Without this join, speaker leakage cannot be ruled out. The official
  HF repo is gated (401) and the Mozilla Data Collective needs an account/API
  key/terms; the current metadata pass uses the public mirror
  `fsicoli/common_voice_21_0` and must be re-checked against the official copy
  when credentials exist.
- **GLOBE_V3 selection result (2026-10-28).** The metadata pass indexed 723,532
  rows across 187 shards with 0 failures. The Common Voice join matched 100% of
  rows (50,951 speakers, 1,102.1 h); 450,835 rows carry a self-reported accent.
  The strict selection is 9,688 clips / 13.83 h pre-trim / 889 speakers / 543
  row groups. US, England, Australian, Canadian, and Indian/South Asian English
  each reach 2.50 h pre-trim (ratio 1.0) with 94–201 train speakers and ≥5
  val/test speakers. Irish reaches only 1.32 h and is **evaluation-only**.
  Selection uses 2.5 h pre-trim because the audio is VAD-trimmed; the
  post-download audit is authoritative for the 2.0 h gate. After excluding 154
  source-clipped clips (1.6%), the audited set is **9,534 clips / 12.28 h
  actual**, with the five trainable classes at **2.185–2.257 h** and 94–198
  train speakers. Irish is 1.176 h actual. No threshold was lowered.
- **Naturalness LoRA corpus (StyleTTS2).** Lowest-friction research path:
  VoxCeleb2, already local, with Whisper pseudo-transcripts; it is
  research/non-commercial only. The measured pilot is 432 clips / 263
  speakers / 0.741 h; see Phase 1.5. Alternatives: AMI (CC BY 4.0, ~100 h),
  Expresso (CC BY-NC 4.0, ~40 h, 4 speakers), or MSP-Podcast after the
  agreement. Do not use Expresso for commercial training.
- **Human MOS/CMOS.** Requires real listeners, informed consent, and likely
  institutional IRB/ethics review. No ratings exist yet; do not fabricate
  them. The current harness reports MOS as `unavailable`.

## 6. Production-grade definition (future gate; not a current claim)

A control is production-grade only when all of the following hold. None of the
current emotion or accent controls meets these gates:

1. **Measurable:** the intended acoustic/prosodic effect is present and larger
   than the neutral variation.
2. **Monotonic:** increasing intensity moves the intended dimension in the
   expected direction without collapsing other dimensions.
3. **Intelligible:** WER stays within a pre-registered budget for the language
   and condition.
4. **Identity-preserving:** speaker similarity to the neutral same-speaker
   reference stays above a pre-registered threshold.
5. **Perceptually validated:** human listeners can identify the intended
   emotion/accent above chance, and MOS/CMOS is measured from real ratings.
6. **Legally trainable:** training and evaluation data are licensed for the
   intended use. VoxCeleb2 remains research/non-commercial only.

No control may be described as “authentic,” “full,” or “production-ready” until
all six gates pass.

## 7. Roadmap

Execution note: the naturalness, emotion, and accent research tracks run in
parallel; their evidence levels, gates, status, and non-claims are consolidated
in [`docs/MASTER.md`](docs/MASTER.md) §5-§6.
The phase order below is a dependency order for product claims, not a ban on
exploratory research. Exploratory adapters stay off the serving path.

### Phase 0 — Scope and truth

- Codify this scope in README, CLI, and demo generation.
- Remove out-of-scope categories from the showcase and public claims.
- Keep legacy code/assets only for reproducibility.
- Add expression-control evaluation metrics with explicit availability.

### Phase 1 — Fix the core

- Improve face→voice identity retrieval and candidate identity consistency.
- Keep candidate diversity, but measure identity preservation per candidate.
- The CFVN structure-preservation experiment is a negative result: it trades
  cross-modal retrieval for face-face identity and is not stable across seeds.
  The canonical CFVN and downstream CVAE/adapter remain unchanged.
- Do not add new expression controls to the product surface until the core gate
  is stable. Exploratory adapters may be trained and evaluated off the serving
  path; see [`docs/MASTER.md`](docs/MASTER.md) §5-§6.

### Phase 1.5 — Naturalness: StyleTTS2 LoRA

- Fine-tune StyleTTS2 with LoRA on natural, conversational speech; keep the
  face→voice identity path frozen and stable during this work.
- Start with the already-local VoxCeleb2 plus Whisper pseudo-transcripts for a
  research-only baseline; evaluate AMI or MSP-Podcast if the baseline is
  insufficient.
- Do not call the result production-ready. Measure WER, speaker similarity,
  prosody, and real human MOS; report `unavailable` rather than inventing
  ratings.
- Keep the identity checkpoints and adapter canonical; LoRA must not silently
  change the identity representation.
- Measured pilot: a 432-clip / 263-speaker / 0.741 h speaker-disjoint VoxCeleb2
  corpus with Whisper pseudo-transcripts; rank-8 LoRA on 25 diffusion-U-Net
  layers (434,184 trainable parameters); all 10 epochs completed with 0 failed
  clips. Validation diffusion loss fell from 0.5980 to 0.5226, but the paired
  25-clip generation evaluation found no WER, speaker-similarity, or F0
  improvement beyond noise. The only Holm-significant change was a small drop
  in energy standard deviation, which is not a naturalness metric. This is a
  null result on the objective proxies, not a serving change. Keep the adapter
  experimental and the canonical CFVN/adapter unchanged; the next step is a
  larger conversational corpus (for example AMI) plus real MOS.

### Phase 2 — Emotion, learned research control

- Acquire CREMA-D as a pilot/evaluation corpus; it is below the 2 h/class gate.
- Use MSP-Podcast, after the institutional agreement, as the learned-control
  training corpus.
- Train on the balanced six: neutral, happy, sad, angry, fear, disgust.
- Add surprise only after securing a commercially usable surprise corpus with
  enough speakers; move whispered to a separate voice-quality track or remove
  it.
- Train a learned emotion conditioning module on the frozen StyleTTS2 style
  space, or a small adapter/LoRA if the data supports it.
- Replace fixed emotion deltas and hand-authored profiles.
- Add emotion-classifier accuracy, WER, speaker similarity, F0/duration/energy
  effects, and human MOS/CMOS.

### Phase 3 — Accent, learned research control

- Select a balanced accent subset from GLOBE_V3 after recovering speaker IDs
  from Common Voice metadata and counting speakers per accent.
- Do not ship accents below 30 speakers / 2 hours; keep them evaluation-only.
- Replace regex respellings with accent-aware G2P/pronunciation lexicons.
- Train a learned accent conditioning module with identity preservation.
- Add accent-classifier accuracy, WER, speaker similarity, and human
  identification tests.
- Treat accent as a user-selected style, never as a face-derived attribute.

## 8. Legacy policy

- Out-of-scope modules may remain importable for tests and historical
  reproducibility.
- Out-of-scope CLI flags and showcase categories must be removed from the
  default product path.
- Out-of-scope generated assets may be archived; they must not appear in the
  default demo report or README capability table.
- No new work, bug fixes, or claims for out-of-scope features unless they are
  needed to keep the in-scope path stable.
