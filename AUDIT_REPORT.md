# VOIX Face-to-Voice System — Code-Oriented Audit

**Date:** 2026-10-24
**Scope:** current working tree (including uncommitted changes), source and artifacts only; documentation treated as non-authoritative. No code was modified during the audit.

**Verdict:** The system is not production-ready, and the central product claim — that a face produces a distinct, identity-consistent voice — is not currently supported by the artifacts or the runtime behavior. Three independent stages destroy the conditioning signal, the training/evaluation protocol cannot detect this, and several user-facing controls are cosmetic or broken. The good news is that the raw data still contains identity information, so this is recoverable with a representation-contract rewrite and retraining, not a data recollection problem.

---

## 1. Method and evidence basis

Evidence was gathered from:

- Static reading of the full pipeline, model, training, data, CLI, service, evaluation, and test code.
- Runtime probes with the project venv (`D:\voix\.venv\Scripts\python.exe`) against the actual checkpoints and HDF5 artifacts.
- Fresh probes that supersede several earlier numerical claims. In particular, the raw HDF5 fused features are **also degenerate**: at cosine >0.99 they collapse into 11 cross-speaker clusters, and the old extraction reused a handful of interviewers/background faces across speakers. The CFVN collapse is real, but it sits on top of a broken face corpus.

Evidence labels used below:

- **Confirmed (runtime):** reproduced by executing code against real artifacts.
- **Confirmed (code):** directly visible in source with no ambiguity.
- **Hypothesis:** plausible cause not yet isolated by experiment.

---

## 2. Actual architecture vs. intended architecture

The intended story is:

```
face image → 560-D fused biometrics → CFVN → 192-D voice identity
           → CVAE → diverse 192-D speaker embeddings
           → adapter → 256-D StyleTTS2 style
           → StyleTTS2 → waveform
```

The actual dataflow has at least four incompatible feature contracts:

```
IMAGE PATH
  image → ArcFace 512 + FaceMesh 32 + Demo 16
        → per-modality L2 normalize
        → concat 560 (norm ≈ 1.732)
        → z-score with face_scaler.pt (n_train=55,448)
        → CFVN → 192 → CVAE → 192 → adapter → 256
        → biometric prosody overwrite (80%)
        → StyleTTS2 → peak normalize to 0.90

HDF5 / PRESET PATH
  features.h5 (80,832×560, built with an untrained FaceFusionLayer)
        → raw (or inconsistently scaled) 560
        → CFVN → same downstream

CFVN TRAINING
  clean_index.csv fused features → z-scored → CFVN
  best checkpoint: epoch 2, val_cos ≈ 0.12883

CVAE / ADAPTER TRAINING
  features.h5 raw fused → CFVN(raw) → embeddings
  random clip-level split
  CVAE target: ECAPA 192
  adapter target: nearest-neighbor timbre/prosody from 497 style pairs

SERVING
  adapter input = CVAE decoder output, not CFVN(raw)
```

This is the structural root cause of most downstream failures. Each stage was trained on a different definition of “face feature,” and the serving path uses yet another.

---

## 3. Severity-ranked findings

### S0 — Core claim is not delivered

#### F-01. CFVN collapses face identity into a narrow cone; cross-modal retrieval is at chance

**Confirmed (runtime).**

On 2,000 clips with correct metadata speaker labels:

| Metric | Raw fused features | CFVN output |
|---|---:|---:|
| Within-speaker cosine | 0.9982 | 0.999999 |
| Between-speaker cosine | 0.2907 | 0.855 |
| Pairwise cosine | — | 0.855 |
| Per-dim std | — | 0.0268 |

Cross-modal face→voice retrieval on 2,000 clean-index val clips:

| Metric | CFVN |
|---|---:|
| Mean ground-truth cosine | 0.1290 |
| R@1 | 0.0015 (0.15%) |
| R@10 | 0.0280 (2.8%) |
| Constant baseline R@1 | 0.0020 (0.20%) |
| Shuffled baseline R@1 | 0.0025 (0.25%) |

Face-to-face identity retrieval (500 speakers, one query/gallery clip each):

| Representation | R@1 | R@10 |
|---|---:|---:|
| Raw fused | 0.0640 | 0.3040 |
| Z-scored raw | 0.0620 | — |
| CFVN | 0.0360 | 0.2680 |

**Corpus-level caveat (runtime).** The raw fused features are not a usable identity representation either. On a 3,000-clip sample, greedy connected components at cosine >0.99 collapse into 11 large clusters plus one singleton. Each large cluster contains 67–266 distinct `clean_index` speakers; the largest clusters contain 373, 369, 369, 357, 341, 341, 305, 212, 139, 101, and 92 clips. 10.1% of all clip pairs exceed 0.99, including 10.0% of cross-speaker pairs. The within-speaker mean of 0.998 is therefore not evidence of identity: it reflects near-duplicate detections of the same few faces (likely interviewers or background people) reused across speakers. The HDF5 `speaker_ids` dataset also has only 619 unique values for 4,978 metadata speakers. Raw face→face R@1 of 6.4% is above chance but far too low for a 500-speaker gallery, and the feature space is dominated by 11 identities. Re-extraction from raw video with speaking-face selection is required; the old fused artifact must not be used for training.

CFVN **reduces** face identity discrimination relative to the raw representation. The checkpoint history is consistent: best val_cos ≈ 0.12883 at epoch 2, with later epochs degrading.

**Code:** `voix/models/contrastive_net.py:23-53` (LayerNorm MLP + L2 normalize); `voix/inference/pipeline.py:350-352` (single CFVN projection).

**Root cause:** Confirmed collapse. The precise training cause is a hypothesis: the combination of LayerNorm, L2-normalized outputs, InfoNCE with in-batch negatives, and checkpoint selection on a weak val_cos metric likely rewards a low-variance cone. Early best epoch 2 suggests the objective degrades identity quickly. The raw-vs-z-scored input mismatch is currently masked by LayerNorm (CFVN output cosine ≈ 0.99996 between the two), so it is a latent contract bug rather than the immediate cause.

**Impact:** Everything downstream is conditioned on an embedding that is nearly the same for every face. CVAE, adapter, prosody, and metadata all inherit this. No amount of CVAE/adapter tuning can recover identity that CFVN removed.

**Fix direction:**

1. Retrain CFVN on a speaker-disjoint split with the exact serving preprocessing.
2. Replace or augment the objective: supervised contrastive with speaker labels, margin-based losses, or a regression/classification head with an explicit identity-preservation term.
3. Select checkpoints on retrieval metrics (R@1/R@10, EER), not mean cosine.
4. Monitor within/between cosine and embedding variance every epoch.
5. Consider removing CFVN entirely and learning the face→voice mapping jointly with the CVAE, or using a frozen face-recognition embedding plus a learned calibration.

**Validation criteria:**

- Held-out-speaker cross-modal R@1 at least 10× chance and materially above constant/shuffled baselines.
- Between-speaker cosine in a healthy range (not >0.8), within-speaker cosine high but not degenerate.
- CFVN face-face R@1 ≥ raw face-face R@1 (no identity loss).
- Same-subject image vs. HDF5 embedding cosine > 0.95 after canonical preprocessing.

---

#### F-01b. The fusion contract silently discards ArcFace identity; V2 per-modality normalization fixes it

**Confirmed (runtime).**

The old `FaceFusionLayer` concatenated ArcFace 512 (unit norm), craniofacial 32 (ratios ~0.5–5.0), and demographics 16 (0–1), then applied a fixed seed-42 random projection. The geometry/demographics dominate the input variance, so the projected vector is nearly identical for every face. On the 100-speaker / 382-clip V2 pilot:

| Representation | Face→face R@1 | Within-speaker cosine | Between-speaker cosine |
|---|---:|---:|---:|
| Raw ArcFace 512 | 0.9424 | 0.5292 | 0.0071 |
| V2 fused 560 (per-modality L2) | 0.9031 | 0.8571 | — |
| V1 fused 560 (no normalization) | 0.0942 | 0.9744 | 0.9754 |
| ECAPA voice 192 | 0.9398 | 0.5183 | 0.0206 |

The V1 fused features are worse than raw ArcFace by an order of magnitude and are essentially constant across speakers. The fix is to L2-normalize each modality before concatenation; `FaceFusionLayer.forward` now does this, and the V2 extraction stores both the raw ArcFace 512 vector and the fixed 560-D fused vector. The V2 fused representation is healthy enough to train on, while raw ArcFace remains the strongest diagnostic baseline.

The same bug affected preset generation: `PresetRegistry` called the fusion layer with unnormalized synthetic geometry, so preset vectors were in the same collapsed space. Regenerating the preset `.pt` files with the fixed layer is required.

**Code:** `voix/models/fusion.py:forward`; `voix/data/feature_contract.py`; `voix/data/presets.py:_generate_synthetic_face_vector`; `scripts/extract_voxceleb2_v2.py`.

**Validation criteria:** V2 fused face→face R@1 ≥0.85 on held-out speakers; raw ArcFace face→face R@1 ≥0.90; image/HDF5 parity max abs deviation <1e-4.

---

#### F-02. CVAE decoder collapses; the decoder diversity loss has the wrong sign and actively encourages collapse

**Confirmed (runtime + code).**

| Quantity | Value |
|---|---:|
| `prior_mu` pairwise cosine | 0.0132 |
| Decoder output pairwise cosine | 0.9805 |
| Decoder within-face candidate cosine | 0.8886 |
| Decoder across-face candidate cosine | 0.8870 |
| Decoder per-dim std (sampled z) | 0.0241 |

The prior is diverse. The decoder maps different latents and different faces to nearly the same output.

**Code:** `voix/training/trainer_cvae_v3_diversity.py:93-100` returns `-off_diag.mean()` with the comment “Negative because we want to MINIMIZE similarity.” At line 203 the loss is **added**:

```python
loss = total_loss + LAMBDA_NCE * nce + LAMBDA_FS * fs_loss + LAMBDA_DIV * div_loss
```

Minimizing `-mean(similarity)` maximizes similarity. The loss is doing the opposite of its name.

**Second confirmed defect:** validation and best-checkpoint selection use only reconstruction + KL (`trainer_cvae_v3_diversity.py:222-230`), excluding NCE, face-separation, and diversity. The selected checkpoint (epoch 150, val_loss 0.6916768) is therefore chosen by a criterion that favors collapse.

**Third confirmed defect:** `CVAE.sample_voices` crashes for batched input:

```
RuntimeError: The expanded size of the tensor (5) must match the existing size (200)
```

`voix/models/cvae.py:323-324` uses `mu_prior.expand(num_samples, -1)`, which only works for a single face. The pipeline currently calls it with a `(1, 192)` input, so the batch bug is latent, but it blocks batched inference and any future multi-face use.

**Impact:** Candidate voices are near-identical regardless of seed or temperature. “Diversity” in the product is not real.

**Fix direction:**

1. Correct the sign: either return `+off_diag.mean()` and subtract it, or keep the negative return and subtract the term.
2. Add decoder output variance/covariance regularization, not just pairwise cosine.
3. Include NCE, face separation, and diversity in validation and checkpoint selection.
4. Add a posterior-collapse diagnostic: mutual information / correlation between prior samples and decoder outputs.
5. Fix `sample_voices` batching with `repeat`/`expand` on the correct leading dimension.

**Validation criteria:**

- Decoder pairwise cosine below an agreed threshold (e.g., <0.5, ideally <0.3) on held-out faces.
- Within-face candidate cosine > across-face cosine by a clear margin (e.g., >0.1).
- Candidate F0/style spread comparable to the style-pair target distribution.
- Different faces produce measurably different decoder outputs; same face with different seeds produces controlled variation.
- Batched `sample_voices` works for B>1 and matches per-item results.

---

#### F-03. Style adapter is near-constant and was trained on a different input distribution than it is served

**Confirmed (runtime + code).**

| Input to adapter | Pairwise cosine | Per-dim std |
|---|---:|---:|
| CFVN embeddings | 0.99958 | 0.00114 |
| CVAE decoder outputs | 0.99994 | 0.00048 |

Both timbre and prosody halves are near-constant.

**Code:** `trainer_adapter_cfvn.py:76` computes CFVN embeddings from raw `features.h5`; `pipeline.py:431` serves the adapter from `e_s_raw_k`, which is the CVAE decoder output. The trainer comment/design claims parity, but the serving distribution is different.

**Additional confirmed weakness:** adapter targets are nearest-neighbor matches from only 497 style pairs. ECAPA pair cosine is ≈0.151, timbre pair cosine ≈0.716, prosody pair cosine ≈0.514, and only 186 of 497 ECAPA vectors are nearest neighbors of any other. Target coverage is thin and noisy.

**Impact:** The adapter contributes almost no face-specific timbre or prosody. The later biometric prosody overwrite then supplies a deterministic gender/age vector, which is why same-gender/same-age faces sound similar.

**Fix direction:**

1. Train the adapter on exactly the serving distribution: sample CVAE outputs, including stochasticity, and use those as inputs.
2. Add a variance/diversity term and monitor output statistics.
3. Improve target construction: more style pairs, better matching, or direct regression from speaker embeddings to style statistics.
4. Consider predicting a residual from a mean style rather than absolute style, with normalization.
5. Evaluate adapter output discrimination with ABX or retrieval, not just MSE.

**Validation criteria:**

- Adapter output pairwise cosine below threshold (e.g., <0.9; target <0.7).
- Per-dim std materially above the current 0.001.
- Timbre/prosody MSE on held-out speakers, plus ABX discrimination above chance.
- Same face → similar style; different faces → measurably different style.
- Train/serve input distribution test: KS/cosine comparison between training and serving embeddings.

---

#### F-04. Four incompatible feature representations; scaling is heuristic and path-dependent

**Confirmed (code + runtime).**

| Path | Representation | Scaling |
|---|---|---|
| Image inference | L2-normalized per-modality concat (norm ≈1.732) | z-score with `face_scaler.pt` |
| HDF5 training | `features.h5` raw 560-D from untrained `FaceFusionLayer` | raw for CVAE/adapter; z-scored for CFVN training |
| Preset via `generate(str id)` | untrained fusion output | z-score applied |
| Preset via `generate_from_preset` | untrained fusion output | raw passed; tensor branch skips scaling because `abs().mean() < 1.5` |
| CFVN training | clean_index fused | z-scored |
| CVAE/adapter training | HDF5 raw | raw |
| Adapter serving | CVAE decoder output | no scaler |

**Code:** `pipeline.py:290-348` (path branches and the `abs().mean() > 1.5` heuristic); `pipeline.py:226-235` (image path bypasses `FaceFusionLayer`); `pipeline.py:299-300` vs. `pipeline.py:334` (preset scaling inconsistency); `trainer_cvae_v3_diversity.py:130-142` and `trainer_adapter_cfvn.py:62-78` (raw HDF5 into CFVN).

**Runtime evidence of divergence:**

- Image z-score vector norm ≈14.44, mean abs ≈0.5395, max abs ≈2.29.
- Nearest HDF5 raw cosine ≈0.0878; nearest HDF5-z cosine ≈0.0960.
- Image z vs. nearest HDF5-z absolute deviation: mean ≈0.87, max ≈8.73.
- CFVN image vs. HDF5 output cosine ≈0.8850 — high only because CFVN’s cone collapses everything.

**Impact:** Silent distribution shift. Retraining any one stage can invalidate the others. The `abs().mean()` heuristic is not a contract; it is a guess that happens to route current tensors differently.

**Fix direction:**

1. Create one canonical feature builder module with an explicit versioned schema: raw modalities, fusion method, scaler version, normalization order.
2. Store both raw and normalized representations, plus metadata identifying the contract.
3. Remove the `abs().mean()` heuristic; require an explicit `representation=` argument or typed wrapper.
4. Make presets and images use the same builder and scaler.
5. Put the preprocessing contract inside checkpoints and validate at load time.
6. Add golden-vector tests for each path.

**Validation criteria:**

- Same subject through image path and HDF5 path yields cosine >0.95 after canonical preprocessing.
- Mean/std/norm of embeddings match within tight tolerance across paths.
- Preset path and image path produce statistically indistinguishable CFVN input distributions.
- No code path can silently skip scaling.

---

### S1 — Major correctness and product-integrity defects

#### F-05. Random clip-level splits leak speakers; reported validation is not generalization

**Confirmed (runtime).**

| Trainer | Val speakers seen in train | Val clips from seen speakers |
|---|---:|---:|
| CVAE | 3,754 / 3,759 | 8,078 / 8,083 |
| Adapter | 4,275 / 4,285 | 12,109 / 12,124 |

The repository already contains a speaker-disjoint split, `clean_index.csv`:

| Split | Clips | Speakers |
|---|---:|---:|
| Train | 55,448 | 3,029 |
| Val | 7,161 | 400 |
| Test | 9,075 | 500 |
| Total | 71,684 | 3,929 |

No train/val or train/test speaker overlap. HDF5 contains 80,832 clips / 4,978 speakers, so 9,148 clips are outside the clean index, and both trainers use all 80,832.

**Code:** `trainer_cvae_v3_diversity.py:150-154` (`random_split`); `trainer_adapter_cfvn.py:107-118` (`torch.randperm`).

**Impact:** Validation losses and checkpoint selection are optimistic and cannot measure unseen-face generalization. The reported CVAE val_loss 0.6916768 and adapter val_loss 0.0327937 are not credible generalization estimates.

**Fix direction:** Use `clean_index.csv` speaker-disjoint train/val/test for every stage. Filter HDF5 to the clean index or document and evaluate the extra 9,148 clips separately. Never split clips randomly when the claim is face generalization.

**Validation criteria:** Assert zero speaker overlap programmatically in the data loader. Report clip and speaker counts per split. Evaluate all models on the same held-out speaker set.

---

#### F-06. Craniofacial sliders are no-ops or actively wrong in every entry point

**Confirmed (code + runtime).**

1. **Production CLI uses invalid keys.** `voix/cli.py:88-94` sets `face_height_to_ioc`, `jaw_width_to_ioc`, `face_width_to_ioc`, `mandible_angle_deg`. `apply_craniofacial_sliders` silently ignores unknown keys, so these sliders do nothing.

2. **`raw_e_geo` NameError.** `voix/inference/cli.py:916` and `:986` reference `raw_e_geo` in `main()`’s candidates/single branches, but it is only defined locally inside `run_showcase_suite` at `:382`. Those CLI modes crash.

3. **Showcase morphology is always `None`.** `resolve_face_input` returns `comp` without a `"morphology"` key, so `:382` sets `raw_e_geo=None`. Morphology sliders in the showcase are ineffective.

4. **Showcase passes z-scored morphology as raw ratios.** `run_showcase_suite` passes `e_f_norm` into `pipeline.generate` with `e_geo=None`. The tensor branch then treats `e_f[:,512:544]` — z-scored morphology — as raw ratios and applies `apply_craniofacial_to_style`. This produces invalid modulation for every showcase asset.

5. **Craniofacial mapping is miscalibrated.** `craniofacial_style.py` assumes neutral = 1.0 for all ratios, but actual ratios are e.g. face height ≈1.75–2.2, jaw ≈1.28–1.4, lips ≈0.18/0.22, chin ≈0.05. Even a neutral face produces large spurious deltas, and the sign contradicts the comments.

6. **Morphology duplicates.** Indices 1 and 7 both bizygomatic/IOC; 0/24 related; z-depth indices 30/31 are not IOC-normalized; the `face_width` slider targets duplicated indices 1/7/15/16, multiplying the same geometric change.

**Impact:** The flagship “craniofacial control” feature is non-functional or harmful. In showcase mode it may degrade every sample.

**Fix direction:** Define a single slider-key schema shared by CLI, pipeline, and morphology module. Pass raw ratios explicitly, never infer them from a z-scored slice. Calibrate neutral values from dataset percentiles, not 1.0. Deduplicate the 32-D layout and document each index. Add a test that every CLI slider changes the output.

**Validation criteria:** For each slider, output style/morphology changes monotonically and only in the intended direction; neutral slider values produce zero delta; unknown keys raise, not silently ignore; showcase and production paths produce identical results for the same raw input.

---

#### F-07. Biometric prosody overwrite erases adapter, emotion, accent, and craniofacial prosody

**Confirmed (code).**

`tts_backend.py:185-264` replaces 80% of the prosody half with a deterministic vector derived only from gender and age:

```python
blended = 0.20 * ref_s[:, 128:] + 0.80 * prosody_bias
```

Same gender + same age → identical prosody. Accent, emotion, and craniofacial style deltas applied earlier in `pipeline.py:442-458` are largely erased.

**Impact:** Individuality and most expressive controls are lost at the final conditioning step. This explains why demo emotion F0 values are nearly identical (neutral 182.1, happy 177.8, sad 180.9, angry 181.1, whispered 183.5, surprised 180.0 Hz) and durations are all ~7.5–7.6 s.

**Fix direction:** Replace the overwrite with a bounded residual bias added to the adapter prosody, preserving most of the adapter signal. Use continuous F0 targets derived from face/age, not a hard gender switch. Apply emotion/accent deltas after the bias, or make the bias part of the adapter training target. Add an explicit `prosody_bias_strength` with a low default.

**Validation criteria:** Same gender/age but different faces produce different F0/prosody. Emotion conditions produce significant F0/energy/duration effect sizes. Accent conditions change pronunciation measurably. Ablation shows adapter prosody still contributes.

---

#### F-08. TTS backend ignores most of its documented control contract, silently falls back, and mislabels normalization

**Confirmed (code + runtime).**

`tts_backend.py` accepts but does not use `f0_scale`, `f0_variance_scale`, `target_f0`, `loudness_lufs`, `blend_weight`, `anchor_path`, and `duration_scale` on the neural path. `_lufs_normalize` is defined at `:31` but never called. The neural path peak-normalizes to 0.90 at `:353-357` despite the comment claiming EBU R128.

**Update (2026-10-28):** the neural path now calls `_lufs_normalize()` and uses a -1 dBTP true-peak limiter; `save_audio_sample()` no longer peak-normalizes. See §10.9. The parameter-contract and silent-fallback findings still stand.

Any exception in neural synthesis is caught at `:360-363`, printed, and the code falls through to `_synthesize_fallback` at `:366`. Tests force `use_neural=False` via `"pytest" in sys.modules`, so fallback success is reported as test success.

**Impact:** Users believe they are controlling pitch, loudness, duration, and anchoring when they are not. A model/runtime failure silently produces a tone-like fallback instead of an error.

**Fix direction:** Implement or remove each parameter. Call `_lufs_normalize` on the neural path. Fail loudly (or return an explicit degraded status) instead of silent fallback. Add a `backend` field to outputs. Make tests skip neural tests when the model is unavailable, not silently substitute fallback.

**Validation criteria:** Parameter sweep shows measurable F0/duration/LUFS changes. LUFS output within tolerance of target. Neural failure raises or is explicitly marked. Integration test uses real StyleTTS2 and skips, not passes, when unavailable.

---

#### F-09. Emotion, accent, cross-lingual, and speech-difference controls are largely cosmetic

**Confirmed (code + metadata).**

- `emotion_mods["energy_scale"]` is never applied.
- “Whispered” does not produce unvoiced whisper; it only adjusts modest prosody/duration/beta.
- Accent control is hardcoded word respellings independent of strength (except an early return).
- Cross-lingual is English G2P/respelling, not true multilingual TTS.
- Demo metadata shows near-identical F0/duration across emotions.

**Impact:** The feature matrix overstates capability. Users cannot rely on these controls for accessibility, accent, or emotion use cases.

**Fix direction:** Either implement real acoustic/prosodic controls (energy envelope, unvoiced excitation, phoneme-level duration, language-specific G2P/TTS) or clearly scope them as “stylistic hints.” Add objective tests: F0 mean/std, energy, duration, ASR WER for accent, and human/listener validation for emotion.

**Validation criteria:** Each control has a measurable, reproducible acoustic effect; strength is monotonic; no control is a no-op; claims match measured behavior.

---

#### F-10. API/service crashes and path-handling inconsistencies

**Confirmed (code).**

- `voix/inference/cli.py:916,986`: `raw_e_geo` NameError in candidates/single modes.
- `CVAE.sample_voices` batch crash (`cvae.py:323-324`).
- `voix/service.py:74-78` rejects preset IDs because it requires `Path(face_image).exists()`, even though the inference CLI supports presets.
- `VoixService.save_all` (`service.py:139-154`) treats a `.wav` destination as a directory when multiple candidates are generated, writing `voice_candidate_1.wav` inside a path that looks like a file.
- `VoixService` assumes `raw["waveform"]` is a torch tensor and calls `.numpy()` (`service.py:218`); currently true, but brittle if the pipeline returns NumPy.

**Impact:** Common CLI modes crash; preset service integration is impossible; multi-candidate saves produce surprising paths.

**Fix direction:** Define a single `FaceInput` type that accepts path/preset/tensor and validates accordingly. Fix `raw_e_geo` scoping. Fix `sample_voices` batching. Make `save_all` handle file vs. directory explicitly. Normalize waveform types at the pipeline boundary.

**Validation criteria:** CLI candidates/single/showcase modes run end-to-end; preset IDs work through the service; `save_all("out.wav")` with N>1 either errors clearly or writes a documented set; waveform type contract tested.

---

#### F-11. `FaceFusionLayer` is untrained and missing; image inference bypasses it

**Confirmed (runtime + code).**

`checkpoints/face_fusion.pt` is missing. HDF5 and presets use an untrained deterministic `FaceFusionLayer` seeded at 42. Image inference bypasses fusion entirely and uses L2-normalized concatenation (`pipeline.py:226-235`).

**Impact:** The 560-D “fused” representation has no learned cross-modal interaction. Image and HDF5 features are structurally different. Any claim of “learned fusion” is false.

**Fix direction:** Either train and version a fusion layer on speaker-disjoint data, or remove it and standardize on the deterministic concat everywhere. Do not keep an untrained module in the artifact path.

**Validation criteria:** One fusion implementation used by all paths; checkpoint exists and is loaded strictly; ablation shows fusion helps or it is removed.

---

#### F-12. Metadata reports fabricated or incorrect model/feature information

**Confirmed (code).**

- `build_detailed_metadata` interprets `e_demo[0]` as `p_male` and `e_demo[1]` as `p_female`, with age bins starting at offset 2 (`cli.py:285-297`). Actual layout is age bins 0–7, sex 8–9, vocal age 10–13, build 14–15.
- Preset inputs are sliced into fake ArcFace/morphology/demographic components.
- Image metadata reports L2-normalized demo values as probabilities.
- Metadata reports `FaceFusionLayer` fusion even though image inference bypasses it.
- Metadata reports `blend_weight` although `pipeline.py:466` forces `eff_blend_weight = 0.0`.
- Metadata reports “CVAE v2 / 128-D adapter” although the actual stack is CVAE face192/latent128 and a dual-head 256-D adapter (`cli.py:340-341`).

**Impact:** Debugging and user-facing explanations are misleading. The metadata cannot be used to audit the actual run.

**Fix direction:** Derive metadata from the actual loaded objects and preprocessing contract, not hardcoded strings. Fix the demographic layout decoder. Report the true adapter output dimension and whether fusion/blending were used.

**Validation criteria:** Metadata round-trips against runtime object inspection; demographic posteriors sum correctly and match the documented layout; `blend_weight` reflects the effective value; model version strings match checkpoint metadata.

---

#### F-13. Watermarking is instantiated unconditionally and failures are swallowed

**Confirmed (code).**

`AudioSealWatermarker` is constructed even when `enable_watermarking=False`. Watermark failures are caught and set `is_watermarked=False` without failing the request. Tests do not verify detection or confidence.

**Impact:** The watermark guarantee is unverifiable. A silent failure looks like an unwatermarked output, and the user is not told.

**Fix direction:** Lazy-init only when enabled. On failure, either fail the request or return an explicit `watermark_status="failed"`. Add detection tests with confidence thresholds and a negative control.

**Validation criteria:** Enabled → watermark detected with confidence above threshold; disabled → no watermark; failure → explicit error/status, not silent false.

---

### S2 — Evaluation and test integrity

#### F-14. The evaluation suite cannot detect the failures above and in places fabricates results

**Confirmed (code + runtime).**

- `scripts/run_mos_evaluation.py` is not human MOS. It generates scores with formulas and `np.random.uniform` (`:83-84`) and then reports “acceptance criteria met” (`:115,133`).
- `scripts/benchmark.py` has placeholder WER/FAD/MOS-N = -1; `compute_audio_metrics` bypasses CVAE and runs the adapter on arbitrary model output; gender mapping is inverted (`gender_val = 0 if male else 1`) relative to the pipeline.
- `voix/evaluation/metrics.py:219` imports missing `voix.data.face_mesh`, so `face_voice_consistency` returns 0.0.
- The same metric calls CFVN on `e_f_full` without z-scoring and builds features differently from the pipeline.
- `speaker_similarity` reloads the model on every call.
- `_load_wav` returns `target_sr` even when resampling is skipped due to missing `resampy`.
- Hardcoded `D:/voix` paths.

**Impact:** Reported quality numbers are not trustworthy. The MOS script is especially damaging because it presents random numbers as human evaluation.

**Fix direction:** Delete or clearly mark fabricated metrics. Implement a real evaluation harness: speaker-disjoint test set, real WER (Whisper), FAD (or a documented proxy), speaker similarity (ECAPA), retrieval metrics, and human MOS with proper protocol. Make `face_voice_consistency` use the canonical feature builder. Add confidence intervals and negative controls.

**Validation criteria:** Every reported metric is traceable to real audio and a real model; no `np.random` in evaluation; negative controls (shuffled faces, constant voice) score near chance; results reproducible from a clean checkout.

---

#### F-15. Tests give false confidence

**Confirmed (code + runtime).**

- `test_pipeline.py` uses random 560-D vectors and pytest fallback TTS.
- `test_tts_backend.py` passes on harmonic fallback.
- `test_ecapa_diversity.py` runs pipeline synthesis at import/collection time with no assertions.
- `test_demographic_presets.py` only checks shape/distinctness.
- No test verifies CFVN representation parity or retrieval quality.
- The targeted subset passes (39 passed in 11.63 s) but exercises almost none of the real failure modes.

**Impact:** CI is green while the product is broken. The tests actively mask the neural path.

**Fix direction:** Add contract tests (feature schema, path parity), model-quality tests (retrieval thresholds, diversity thresholds), and integration tests marked `neural` that skip when the model is unavailable. Ban random-vector-only tests for core claims. Add regression tests for every confirmed defect in this report.

**Validation criteria:** A deliberately collapsed checkpoint fails the suite. A correct checkpoint passes. Neural tests are skipped, not silently replaced, when dependencies are missing. Coverage includes retrieval, diversity, slider effects, and metadata correctness.

---

#### F-16. Config drift, missing artifacts, permissive checkpoint loading, hardcoded paths

**Confirmed (code + runtime).**

- `configs/environment.yaml` references missing `feature_scaler.pt`, `training_index.csv`, and `checkpoints/face_fusion.pt`.
- Pipeline ignores many config values; `inference.diffusion_steps: 25` vs. CLI showcase default 5 vs. pipeline `_inf.get("diffusion_steps", 5)`. **Update (2026-10-27):** `inference.diffusion_steps` is now `null`, and the pipeline resolves `configs/hardware.yaml` through `get_hardware_profile()`; see §10.9 for the remaining checked-in-demo caveat.
- `face_scaler_v1.pt` exists but is unused.
- `D:/voix` is hardcoded across pipeline, trainers, scripts, presets, and config.
- CVAE/adapter load with `strict=False` and only print missing/unexpected keys (`pipeline.py:102,135`).

**Impact:** The system is not portable or reproducible. Architecture drift can load silently with random weights.

**Fix direction:** Central config with environment-variable/project-root resolution; validate required artifacts at startup; load checkpoints strictly or fail on missing keys; remove dead scaler artifacts; align diffusion steps across config/CLI/pipeline.

---

### S3 — Hygiene and maintainability

- Dead/legacy trainers and scripts remain alongside current ones, increasing the chance of running the wrong entry point.
- Broad `except Exception` blocks in the neural path hide real errors.
- Inconsistent naming (`e_f`, `e_f_norm`, `e_f_voice`, `e_s_raw_k`) makes representation bugs easy.
- Comments frequently contradict code (LUFS vs. peak, “negative because minimize,” “learned fusion”).
- Demo WAVs are currently deleted from the working tree; `demo_metadata.json` remains. Do not restore them without a deliberate regeneration plan.

---

## 4. Cross-cutting root causes

1. **No representation contract.** There is no single source of truth for what a “face feature” is. Each stage invented its own.
2. **No speaker-disjoint evaluation discipline.** The clean split exists but is ignored.
3. **No quality gate on learned embeddings.** Checkpoint selection uses losses that do not measure identity or diversity.
4. **Loss sign and model-selection bugs went untested.** The diversity loss is a one-line sign error with no unit test.
5. **Silent fallbacks and broad exception handling.** Failures become plausible-looking outputs.
6. **Evaluation is disconnected from the claim.** Random MOS and placeholder WER cannot falsify anything.
7. **Claims outran code.** Metadata, docs, and CLI help describe a system that was not built.

---

## 5. What is actually working

- The raw VoxCeleb2 video and the voice side are usable. On the 100-speaker V2 pilot, raw ArcFace face→face R@1 is 0.9424 and ECAPA voice→voice R@1 is 0.9398, both far above chance.
- The old HDF5 fused features are **not** usable as an identity representation: 3,000 clips collapse to 11 cross-speaker clusters at cosine >0.99, and 10% of cross-speaker pairs exceed 0.99. The corpus must be rebuilt from raw video.
- The V2 fusion fix (per-modality L2 normalization) raises fused face→face R@1 from 0.0942 to 0.9031 on the same pilot clips.
- `clean_index.csv` is genuinely speaker-disjoint and covers 71,684 clips / 3,929 speakers, but it describes the old broken face corpus; the new manifest is `data/processed/voxceleb2_v2/index.csv` (37,534 clips / 6,112 speakers).
- The CVAE prior is diverse (pairwise cosine ≈0.0132), so the prior network is not collapsed.
- StyleTTS2 is installed and the demo audio was real neural TTS, not fallback tones.
- The pipeline, CLI, service, and tests are structurally complete enough to iterate on.
- The 497 style pairs provide a starting point, though coverage is thin.

The failures are concentrated in learned projections and the training/evaluation protocol, not in the availability of data or the TTS vocoder.

---

## 6. Prioritized remediation plan

### Phase 0 — Stop false confidence (0–2 days)

1. Remove or clearly mark `run_mos_evaluation.py` output as synthetic; delete the “acceptance criteria met” reporting.
2. Make `benchmark.py` fail if WER/FAD/MOS-N are placeholders; remove inverted gender mapping.
3. Make neural TTS failures raise or return `status="degraded"` instead of silently falling back.
4. Fix the `raw_e_geo` NameError and `sample_voices` batch crash.
5. Fix service preset validation and `save_all` path handling.
6. Add startup validation for required checkpoints/scalers; fail on missing artifacts.
7. Freeze external claims about identity, emotion, accent, craniofacial control, and watermarking until validated.

**Exit criteria:** No fabricated metrics in any report; CLI modes run; missing artifacts fail loudly; tests can distinguish neural from fallback.

---

### Phase 1 — Representation contract and CFVN (1–2 weeks)

1. Define a canonical feature builder with a versioned schema and explicit raw/normalized states.
2. Standardize image, HDF5, preset, training, and serving paths on that builder.
3. Train/validate the fusion layer or remove it.
4. Retrain CFVN on the speaker-disjoint `clean_index.csv` split with the exact serving preprocessing.
5. Replace val_cos checkpoint selection with retrieval metrics (R@1/R@10, EER) and identity-preservation metrics.
6. Add contract tests and a retrieval benchmark with constant/shuffled baselines.

**Exit criteria:** Held-out-speaker cross-modal R@1 ≥10× chance; CFVN does not reduce face-face R@1 vs. raw; image/HDF5 same-subject cosine >0.95; all paths use one scaler.

---

### Phase 2 — CVAE and adapter (1–2 weeks)

1. Fix the decoder diversity loss sign.
2. Include NCE, face separation, and diversity in validation/checkpoint selection.
3. Add variance/covariance regularization and posterior-collapse diagnostics.
4. Fix `sample_voices` batching.
5. Retrain CVAE on the fixed CFVN embeddings with the speaker-disjoint split.
6. Retrain the adapter on the actual serving distribution (CVAE samples), with variance/diversity terms.
7. Improve style-pair coverage or target construction.
8. Remove or drastically reduce the biometric prosody overwrite.

**Exit criteria:** Decoder pairwise cosine <0.5; within-face > across-face margin >0.1; adapter output pairwise cosine <0.9; same face stable, different faces discriminable; candidate diversity visible in F0/style metrics.

---

### Phase 3 — Control correctness and TTS contract (1 week)

1. Unify slider keys and pass raw morphology explicitly.
2. Calibrate craniofacial neutral values from data; deduplicate indices.
3. Fix showcase path to use raw ratios, not z-scored slices.
4. Implement or remove `f0_scale`, `f0_variance_scale`, `target_f0`, `loudness_lufs`, `duration_scale`, `blend_weight`, `anchor_path`.
5. Call LUFS normalization on the neural path.
6. Implement real emotion energy/whisper behavior, accent strength, and language-specific TTS, or scope claims down.
7. Add objective control tests (F0, energy, duration, WER, slider monotonicity).

**Exit criteria:** Every control changes output measurably and monotonically; neutral values produce zero delta; LUFS within tolerance; emotion/accent claims match measurements.

---

### Phase 4 — Evaluation and CI (1 week)

1. Build a speaker-disjoint evaluation harness with real WER, speaker similarity, retrieval, FAD/proxy, and diversity metrics.
2. Add human MOS protocol if MOS is claimed; otherwise remove MOS claims.
3. Add negative controls (constant voice, shuffled faces) to every benchmark.
4. Add contract, quality, and integration tests; mark neural tests and skip when unavailable.
5. Add a CI gate that fails on collapsed embeddings, retrieval below threshold, or missing artifacts.
6. Fix `metrics.py` imports, feature path, resampling, and hardcoded paths.

**Exit criteria:** A deliberately collapsed checkpoint fails CI; a correct checkpoint passes; every reported number is reproducible from a clean checkout; negative controls score near chance.

---

### Phase 5 — Cleanup and portability (ongoing)

1. Remove dead trainers/scripts or move them to `legacy/`.
2. Replace hardcoded `D:/voix` with project-root/config resolution.
3. Load checkpoints strictly; fail on key mismatch.
4. Align config, CLI defaults, and pipeline behavior (especially diffusion steps).
5. Reconcile comments/docs with code or delete misleading claims.
6. Add structured logging of representation versions, checkpoint hashes, and backend used.

---

## 7. Validation protocol and suggested thresholds

These are starting points, not universal constants; calibrate against a healthy baseline.

### Representation parity

- Same subject image vs. HDF5: cosine >0.95.
- Mean/std/norm differences <5%.
- No path may skip scaling; assert on tensor metadata or an explicit representation tag.

### CFVN

- Cross-modal R@1 on held-out speakers: ≥10× chance, and above constant/shuffled baselines by a clear margin.
- Face-face R@1: CFVN ≥ raw.
- Between-speaker cosine: below 0.8 (target 0.3–0.6); within-speaker: high but not >0.9999.
- Per-dim std: materially above 0.0268; monitor collapse ratio.

### CVAE

- Prior pairwise cosine: low (current 0.0132 is healthy).
- Decoder pairwise cosine: <0.5, target <0.3.
- Within-face candidate cosine − across-face cosine: >0.1.
- Candidate F0/style spread comparable to target distribution.
- Batched and single-item sampling agree.

### Adapter

- Output pairwise cosine: <0.9, target <0.7.
- Per-dim std: >0.01.
- Timbre/prosody MSE on held-out speakers, plus ABX above chance.
- Train/serve embedding distribution match (KS/cosine).

### Splits

- Zero speaker overlap; assert in code.
- Report clips/speakers per split.
- All models evaluated on the same held-out speaker set.

### Controls

- Slider monotonicity and direction; neutral = zero delta.
- Emotion: significant F0/energy/duration effect sizes; whisper unvoiced.
- Accent: measurable pronunciation change and WER impact.
- LUFS: within ±1 LU of target.
- Watermark: detection confidence above threshold; negative control clean.

### Evaluation integrity

- No `np.random` in evaluation.
- Negative controls near chance.
- Confidence intervals for all metrics.
- Reproducible from a clean checkout.

---

## 8. Confirmed vs. hypothesis summary

| Claim | Status |
|---|---|
| CFVN collapses between-speaker variation; retrieval near chance | **Confirmed (runtime)** |
| Raw HDF5 features retain identity | **Refuted (runtime)** — 11 cross-speaker clusters at cosine >0.99; 67–266 speakers per cluster |
| HDF5 `speaker_ids` match metadata | **Refuted (runtime)** — 619 unique IDs vs 4,978 metadata speakers |
| V2 fused features retain identity | **Confirmed (runtime)** — face→face R@1 0.903 on 100 speakers |
| Raw ArcFace features retain identity | **Confirmed (runtime)** — face→face R@1 0.942 on 100 speakers |
| ECAPA voice features retain identity | **Confirmed (runtime)** — voice→voice R@1 0.940 on 100 speakers |
| CVAE prior diverse, decoder collapsed | **Confirmed (runtime)** |
| Decoder diversity loss sign is inverted | **Confirmed (code)** |
| CVAE checkpoint selection ignores diversity terms | **Confirmed (code)** |
| Adapter near-constant | **Confirmed (runtime)** |
| Adapter trained on CFVN(raw), served CVAE output | **Confirmed (code)** |
| Random clip-level split leaks speakers | **Confirmed (runtime)** |
| Image/preset/HDF5/CFVN/CVAE representations differ | **Confirmed (code + runtime)** |
| Craniofacial sliders are no-ops or wrong | **Confirmed (code)** |
| `raw_e_geo` NameError | **Confirmed (code)** |
| Biometric prosody overwrite erases conditioning | **Confirmed (code)** |
| TTS parameters ignored; silent fallback | **Confirmed (code)** |
| MOS script fabricates scores | **Confirmed (code)** |
| Benchmark placeholders and inverted gender | **Confirmed (code)** |
| `face_voice_consistency` broken | **Confirmed (code)** |
| Tests run fallback and lack assertions | **Confirmed (code + runtime)** |
| Metadata demographic layout wrong | **Confirmed (code)** |
| Watermark failures swallowed | **Confirmed (code)** |
| CFVN collapse caused specifically by InfoNCE/LayerNorm/early stopping | **Hypothesis** — collapse confirmed, cause not isolated |
| Style-pair scarcity causes adapter collapse | **Hypothesis** — plausible contributor, not proven |
| Craniofacial mapping sign contradicts comments | **Confirmed (code)**; perceptual impact not measured |
| Demo emotion F0 similarity is caused by prosody overwrite | **Hypothesis** — consistent with code, not ablated |

---

## 9. Bottom line

The system’s core identity signal is destroyed by CFVN before the generative stages run; the CVAE decoder then collapses what remains; the adapter is near-constant; and the prosody overwrite removes most of the residual individuality. The evaluation stack cannot see any of this because it uses leaked splits, placeholder metrics, fabricated MOS, and fallback TTS tests.

The raw VoxCeleb2 video and the voice side are still good, and the new speaker-disjoint manifest already exists. The old fused face artifact is not usable and must not be reused. The fastest credible path is:

1. **Fix the representation contract and retrain CFVN with retrieval-based selection.**
2. **Fix the CVAE loss sign and selection, then retrain on the fixed CFVN.**
3. **Retrain the adapter on the actual serving distribution and remove the 80% prosody overwrite.**
4. **Replace the evaluation suite with real, speaker-disjoint metrics and negative controls.**

Until Phase 1 and Phase 2 are complete, no claim about face-specific voice identity, candidate diversity, or style control should be treated as validated.

---

## 10. Post-remediation verification

This section records measured results from the rebuilt V2 stack. It is
deliberately separated from the original audit: the findings above describe the
old tree, while everything below was produced after the fixes. Full-corpus
numbers replace the shard-1 smoke numbers once all six dev shards and the
official test shard are extracted.

### 10.1 Rebuilt manifest and shard 1

- Manifest: `data/processed/voxceleb2_v2/index.csv`, 37,534 clips / 6,112
  speakers, speaker-disjoint. Train 33,205 / 5,393, val 3,701 / 601, official
  test 628 / 118. SHA-256
  `f19d09cf0f3f6bd17f9d925cbc6aea57c65efdb8b6a1deefaa6fdfd7db496e53`.
- Shard 1 extraction: 6,527 of 6,870 selected clips (95.0%), 1,118 speakers,
  5,854 train / 673 val. 343 failures: 342 `no_speaking_face`, 1 `no_frames`.
  No speaker lost all clips; maximum failures for one speaker was 5.
- The HDF5 is valid: 6,527 unique clip IDs, no duplicate rows, no zero or
  non-finite feature rows, all 1,118 manifest speakers represented.
- Speaking-face selection quality: 98.2% geometry-ok, 88.5% of clips with at
  least three audio-visual mouth-motion pairs, median track length 15 sampled
  frames. Face-gender attribute accuracy 94.1% (old corpus: 68.5%).

### 10.2 Representation quality (shard 1, held-out val speakers)

| Representation | face→face R@1 | chance | multiple |
|---|---:|---:|---:|
| Raw ArcFace 512-D | 0.9217 | 0.0087 | 106× |
| Raw fused 560-D | 0.8870 | 0.0087 | 102× |
| Z-scored fused 560-D | 0.8870 | 0.0087 | 102× |
| ECAPA voice→voice | 0.9652 | 0.0087 | 111× |

- Shuffled and constant controls scored 0.0087–0.0174, i.e. at chance.
- Z-scored fused health: within-speaker cosine 0.4657, between-speaker
  −0.0004. Raw fused health: within 0.8549, between 0.6985, confirming that the
  raw vector contains a large common component and must not be used as the
  CFVN input without normalization.
- The rebuilt fused representation preserves identity, unlike the old artifact
  (face→face R@1 0.0942, within/between 0.9744/0.9754).

### 10.3 CFVN alignment

The first shard-1 smoke run used the old default `lambda_reg=0.5` and per-clip
ECAPA targets. It preserved face identity (CFVN face→face R@1 0.9043) but
face→voice retrieval stayed at chance (R@1 0.0087) after 10 epochs. This is a
second, independent cause of the old collapse: speaker-classification loss alone
does not align the output with the voice space.

Two changes fixed it:

1. Regress to the L2-normalized **speaker centroid** of the ECAPA embeddings,
   not the noisy paired clip embedding.
2. Raise `lambda_reg` from 0.5 to 5.0 (default) and use `lr=1e-3`.

Shard-1 smoke result with centroid targets:

| Metric | Value |
|---|---:|
| Val face→voice R@1 | **0.1043** (12× chance) |
| Val face→voice R@10 | 0.2696 |
| CFVN face→face R@1 | 0.9043 |
| Raw face→face R@1 | 0.8870 |
| Within-speaker cosine | 0.5743 |
| Between-speaker cosine | 0.1761 |
| Collapse ratio | 0.9031 |

The CFVN now adds cross-modal retrieval above the linear-probe baseline
(0.0435) without reducing face→face identity versus the raw representation.
Full-corpus training will determine whether the 12× margin holds on 601 val
speakers and the 118 official test speakers.

Because the fused 560-D input contains gender and age channels, the
unrestricted face→voice score can be inflated by demographic matching rather
than identity. CFVN evaluation now also reports a gender-controlled score with
the gallery restricted to the query's gender. Chance rises to roughly
`1 / n_same_gender`; a model that only predicts demographics falls to that
floor. The full-corpus fused and raw-ArcFace runs will both record this
diagnostic so the representation choice is not based on a demographic shortcut.

### 10.4 Image/HDF5 parity

Measured on 12 single-track pilot clips by re-reading the raw video, selecting
the stored frame index, re-detecting, re-cropping, re-running MediaPipe and the
demographic estimator, and re-fusing:

| Component | Cosine / IoU |
|---|---:|
| Bounding box IoU | 0.999999 |
| Geometry | 1.0000 |
| Demographics | 0.8901 |
| Raw ArcFace | 0.8790 |
| Fused 560-D | 0.9425 mean / 0.8829 min |

The deterministic preprocessing is exact (bbox and geometry). The fused cosine
is below the audit's original >0.95 target because the HDF5 stores a four-frame
ArcFace average while serving receives a single image. This is a known
limitation, not a silent preprocessing mismatch. Saved annotated frames are not
valid for parity testing because they contain a green bbox/text overlay; the
clean-frame probe is the authoritative measurement
(`data/processed/voxceleb2_v2/parity_probe_cleanframe.json`).

`scripts/measure_serving_parity.py` now automates the same check on a
deterministic, speaker-balanced subset and reports the downstream CFVN
face→face/face→voice effect. It resolves both corpus layouts (the full corpus's
`clips/dev/...` and the pilot's stripped `clips/...`); the first smoke failed
with `frame_unreadable` because the pilot HDF5 stores `dev/`-prefixed clip ids
while the pilot download directory omits the prefix. On five held-out pilot
clips the harness reproduced exact bbox/geometry agreement (IoU 0.999999,
geometry cosine 1.0), ArcFace cosine 0.873, and fused cosine 0.968. The
official-test measurement is deferred until shard 7 is extracted.

### 10.5 Scaler provenance

A scaler that is 560-D and carries the right `face_key` can still have been fit
on an earlier shard. Dimension-only checks therefore cannot detect staleness
after the corpus grows. Scalers now record `schema_version`, `face_key`,
`feature_dim`, `n_train`, and `n_train_speakers`; `load_scaler_payload()`
validates dimension/key and `validate_scaler_for_dataset()` validates the
recorded train rows and speakers.

The three consumers behave differently by design:

- `trainer_cfvn` **rebuilds** an incompatible scaler and records the new
  `n_train`/`n_train_speakers` in the checkpoint.
- `trainer_cvae_v3_diversity` and `trainer_adapter_cfvn` **fail** rather than
  train on a stale normalization.
- `scripts/ci_gate_v2.py` **fails** when the serving scaler's `n_train` or
  `n_train_speakers` does not match the corpus it is gating.

Unit tests cover stale row/speaker counts, wrong key/dimension, the ArcFace
512-D scaler round-trip, and the CFVN reuse/rebuild paths. The earlier CVAE
smoke run that silently used the shard-1 scaler on the 100-speaker pilot corpus
now fails with an explicit `Scaler was fit on 5854 train rows but the corpus
has 363` error.

The eight `data/presets/*.pt` caches had the same problem in a different form:
they were bare 560-D tensors produced by the V1 fusion layer, with pairwise
cosine **0.9997** across all eight presets. The fixed fusion code would still
have loaded those stale tensors, so preset-driven inference would have remained
collapsed. Preset caches are now versioned (`FaceFeatureV2`); bare V1 tensors
are rejected and rebuilt from the canonical definitions. The regenerated
presets have pairwise cosine **0.607–0.796**. The old files are preserved under
`legacy/presets_v1/` for provenance.

### 10.6 Scaler routing for the raw-ArcFace diagnostic

The 512-D diagnostic CFVN needs the separately fit ArcFace scaler. Evaluation
now selects the scaler from the checkpoint's `face_dim`: a 512-D model that is
given the default 560-D path is redirected to
`data/processed/voxceleb2_v2/face_scaler_arcface_v2.pt`, and the generic
`load_scaler_payload(..., expected_dim=512, face_key="arcface")` validates it.
The same path is used by the CI gate. Previously the evaluation helper always
called `load_face_scaler` (560-D), so the diagnostic model could only be
evaluated by silently skipping the metric.

### 10.7 Final V2 verification

The V2 corpus is complete. `data/processed/voxceleb2_v2/features_v2.h5`
contains **35,892 clips / 6,111 speakers** (train 31,749 / 5,392; val 3,539 /
601; official test 604 / 118). One manifest speaker is missing; six speakers
have fewer than two clips. The 1,642 effective failures (4.4%) are all
`no_speaking_face`; there are zero audio-extraction failures. The corpus
validator reports no hard errors. Both scalers were fit on the 31,749 train
rows / 5,392 speakers.

Face-to-face retrieval on held-out speakers:

| Representation | Val R@1 | Test R@1 |
|---|---:|---:|
| Raw ArcFace 512-D | 0.9200 | 0.9322 |
| Raw fused 560-D | 0.8783 | 0.9153 |
| Z-scored fused 560-D | 0.8867 | 0.9153 |
| ECAPA voice-to-voice | 0.9200 | 0.9322 |

Negative controls score 0.0 (shuffled) and chance (constant). Health:
ArcFace within/between 0.5097/0.00696; fused raw 0.8551/0.6996; fused z-scored
0.4647/0.0000324; face-gender attribute accuracy 0.9446. Speaking-face
selection: 98.2% geometry-ok and 88.9% of clips with at least three
audio-visual mouth-motion pairs.

CFVN face-to-voice retrieval on the official test split:

| Input | Best epoch | R@1 | R@10 | Chance multiple | Gender-controlled R@1 | CFVN face-to-face vs raw |
|---|---:|---:|---:|---:|---:|---:|
| Fused 560-D | 20 | **0.0932** | 0.4068 | **11.0x** | 0.0932 (chance 0.0169) | 0.8475 vs 0.9153 |
| Raw ArcFace 512-D | 16 | **0.1017** | 0.4153 | **12.0x** | 0.1017 (chance 0.0169) | 0.8898 vs 0.9322 |

The ArcFace/fused test R@1 ratio is **1.0909**, below the pre-registered
`SWITCH_FACTOR=1.5`; the serving representation remains fused 560-D. The
CFVN's face-to-face identity preservation is lower than raw face-to-face
retrieval, as expected for a 192-D cross-modal bottleneck; the gate checks
embedding health separately.

**Structure-preservation experiment (negative, not adopted).** A raw-face
similarity-distillation term (`--lambda-struct`) plus a stronger face-face
selection weight was tested to close the 0.8475-vs-0.9153 gap. It raises test
face-face R@1 over the s0 baseline in every experimental run, but the result is
not stable:

| Run | λ_struct / FF weight | Val FF vs raw | Test FF vs raw | Test R@1 |
|---|---|---:|---:|---:|
| s0 baseline | 0.0 / 0.10 | 0.8317 vs 0.8783 | 0.8475 vs 0.9153 | **0.0932** |
| s1 | 0.5 / 0.10 | 0.8767 vs 0.8783 | 0.8898 vs 0.9153 | 0.0593 |
| s2 | 1.0 / 0.10 | 0.8750 vs 0.8783 | 0.9068 vs 0.9153 | 0.0593 |
| s3 seed 42 | 0.5 / 0.50 | 0.8867 vs 0.8783 | **0.9153 vs 0.9153** | **0.0932** |
| s3 seed 43 | 0.5 / 0.50 | 0.8900 vs 0.8783 | 0.8814 vs 0.9153 | 0.0678 |
| s3 seed 44 | 0.5 / 0.50 | 0.8933 vs 0.8783 | 0.8983 vs 0.9153 | 0.0678 |
| f1, init baseline, lr 1e-4 | 0.5 / 0.50 | 0.8650 vs 0.8783 | 0.8814 vs 0.9153 | 0.0678 |
| f2, init baseline, lr 1e-4 | 1.0 / 0.50 | 0.8700 vs 0.8783 | 0.8898 vs 0.9153 | 0.0508 |

Only the seed-42 s3 run matches raw face-face R@1 on test, and it does not
improve cross-modal retrieval. Seeds 43 and 44 pass the validation identity
gate (0.8900/0.8933 vs 0.8783) and fail on test (0.8814/0.8983 vs 0.9153).
Under the fixed validation-only rule, seed 44 is selected and
`test_identity_gate_winner` is **false** (`results/cfvn_structure_comparison.json`).
Fine-tuning from the canonical checkpoint at `lr 1e-4` does not fix this.
The structure term is therefore a measured trade-off, not a stable fix:
`results/checkpoints/cfvn_v2_best.pt` remains canonical, and the downstream
CVAE/adapter were not retrained or re-gated.

CVAE (best epoch 51, early stop 81): validation face-to-voice R@1 **0.0383**
(23x chance), R@10 **0.1850**, val loss 4.48136. Decoder within/across cosine
0.5245/0.1325 (margin 0.392), prior across 0.0173 (margin 0.602), 128/128
active units. On the official-test gate: decoder across 0.0449, face margin
0.2229, prior across 0.0329, 128 active units.

Adapter (best epoch 18; 5-clip per-speaker StyleTTS2 targets): validation
prior-mean raw/centered across cosine **0.8537/0.0028**, target cosine 0.8794
(timbre 0.9416, prosody 0.9054), output norm 3.0198 vs target 3.0199, centered
norm 1.3762 vs target 1.3138, per-dim std 0.0204. On the official test split,
CVAE-sample raw/centered across cosine **0.8446/-0.0022**, target cosine
0.8742 (timbre 0.9426, prosody 0.9072), output norm 2.9450, centered norm
1.3735, per-dim std 0.0207. The target's own raw across cosine is 0.8128 and
its centered across cosine is 0.0013; the adapter preserves the common style
mean instead of collapsing to it.

The CI gate now checks the actual cross-modal face-to-voice task (an earlier
version compared face embeddings to face embeddings), adapter target fidelity,
and the full official test split by default. Full-split result:
`results/ci_gate_v2.json` passes all checks - R@1 0.0932, 11.0x chance,
shuffled control 0.0, constant control at chance, adapter across 0.8430,
per-dim std 0.0208, target cosine 0.8736. The archived collapsed adapter
(`legacy/checkpoints/style_adapter_cfvn_best_collapsed.pt`) fails the gate on
adapter across cosine 0.9554 and target cosine 0.8037. A 400-clip smoke subset
(`--max-faces 400`) scores R@1 0.0763 (9.0x chance) and fails the 10x
chance-multiple threshold; this is a query-count power limitation, not a
threshold change, so the full split is the default.

Remaining verification: serving parity on official test clips and human MOS.
Human MOS remains `awaiting_human_ratings` until ratings are supplied.

### 10.8 Scope, demo surface, and expression-control measurement

The active product scope is recorded in `SCOPE.md`: candidate generation is
the core, emotions are the first production-grade expression control, and
accents are the second. Cross-lingual synthesis, speech-difference simulation,
and craniofacial sliders are permanently out of scope. Out-of-scope code and
assets remain for reproducibility but are excluded from the default demo,
README claims, and roadmap.

The v1 demo surface was **15 assets**: 3 candidates, 6 accents, and 6 emotions.
The 14 out-of-scope assets (`04_speech_differences`, `05_cross_lingual`,
`06_craniofacial_sliders`) were moved, not deleted, to
`legacy/demo/out_of_scope_2026-10-27/`. `demo/demo_metadata.json` keeps an
explicit `archived_out_of_scope_assets` inventory, and
`demo/DEMO_SHOWCASE_REPORT.md` was regenerated for the in-scope surface.
`demo/synthesis_manifest.json` covers all 15 assets;
`demo/expression_manifest.json` covers the 12 expression assets and lists the
3 candidate assets separately because candidates are an identity/diversity
condition, not an emotion or accent.

The v1 demo was regenerated on 2026-10-28 after the level fix. The pre-phase-0
snapshot was not reproducible from the current checkpoints despite matching
seed/config; it was moved to
`D:\voix_archive\20261105_cleanup\files\results\demo_pre_phase0` during the
approved 2026-11-05 cleanup. The v1 15 assets were canonical for this
subsection, but the current research demo is v2 (21 clips, 3 voices,
3 emotions, 3 accents) at `results/demo/research_v2/`.

Candidate identity was measured with `scripts/evaluate_candidate_identity.py`
on the same 15 assets (`results/candidate_identity.json`):

| Metric | Measured | Interpretation |
|---|---:|---|
| Candidate face→voice cosine, mean | 0.1019 | weak absolute face-voice link |
| Candidate face→voice cosine, min | 0.0953 | candidate 1 |
| Candidate pairwise ECAPA, mean | 0.7236 | no longer a single collapsed identity |
| Candidate pairwise ECAPA, min | 0.5408 | candidate 1 vs 3 |
| Candidate pairwise ECAPA, max | 0.8688 | candidate 2 vs 3 still close |
| Expression face→voice cosine, mean | 0.0857 | expression variants remain weakly face-consistent |
| Expression identity preservation, mean | 0.5798 | below the 0.80 gate |
| Expression identity preservation, min | 0.2577 (sad) | prosody changes dominate ECAPA |

Reusing the primary candidate's speaker embedding fixes the CVAE identity
sample, but it does not guarantee ECAPA-level identity preservation: the
prosody/style changes themselves move the measured speaker embedding. Candidate
diversity improved relative to the old collapsed demo (1-vs-3 pair 0.5408
versus 0.9393), but candidate 2 vs 3 is still 0.8688 and the absolute
face→voice cosine remains near the CFVN's weak retrieval regime.

Expression controls were measured on the regenerated assets with
`scripts/evaluate_expression_controls.py` (Whisper `base`, CPU, one clip per
condition):

```bash
python scripts/evaluate_expression_controls.py \
  --manifest demo/expression_manifest.json \
  --output results/expression_controls.json \
  --compute-wer --whisper-model base --max-wer 0.20 --device cpu
```

| Condition | F0 mean (Hz) | Energy mean | WER |
|---|---:|---:|---:|
| emotion:neutral | 234.44 | 0.05125 | 0.0000 |
| emotion:happy | 287.02 | 0.05847 | 0.1176 |
| emotion:sad | 211.00 | 0.04682 | 0.1176 |
| emotion:angry | 277.02 | 0.06057 | 0.2353 |
| emotion:whispered | 256.38 | 0.03980 | 0.0000 |
| emotion:surprised | 302.56 | 0.05806 | 0.1176 |
| accent:general_american | 243.95 | 0.05066 | 0.0000 |
| accent:british_rp | 251.37 | 0.05179 | 0.2941 |
| accent:australian | 246.80 | 0.05069 | 0.2941 |
| accent:indian_english | 265.67 | 0.05268 | **0.3529** |
| accent:irish | 256.18 | 0.04942 | 0.1176 |
| accent:scottish | 245.02 | 0.05239 | 0.1176 |

Measured gate result (`results/expression_controls.json`): **fail**.

| Gate | Measured | Threshold | Result |
|---|---:|---:|---|
| Speaker preservation, mean | 0.5092 | >= 0.80 | fail |
| Speaker preservation, min | 0.2364 (sad) | >= 0.80 | fail |
| WER, worst condition | 0.3529 (Indian English) | <= 0.20 | fail |
| WER, 12-condition mean | 0.1471 | - | diagnostic |
| Emotion classifier accuracy | unavailable | >= threshold | unavailable |
| Accent classifier accuracy | unavailable | >= threshold | unavailable |
| Intensity monotonicity | insufficient_data | \|rho\| >= 0.8 | unavailable |
| Human MOS/CMOS | unavailable | - | unavailable |

The emotion-classifier gate stays `unavailable` on the demo assets even though a
CREMA-D pilot classifier now exists: it was trained on acted speech, its
cross-corpus behavior is untested, and there is one demo clip per condition. It
is not a gate and is not used to claim control accuracy.

All 10 speaker-preservation comparisons fall below 0.80; the best is Scottish
vs General American at 0.7737. The accent WER mean is 0.1961 and the emotion
WER mean is 0.0980. Whisper `base` on a single 17-word prompt per condition is
a diagnostic, not a production WER estimate: there are no confidence
intervals, no repeated prompts, and no multi-speaker sweep.

The intended prosody directions hold for happy, angry, and surprised F0
relative to neutral, and for the duration changes. Sad F0 is now 211.00 Hz
versus neutral 234.44 Hz, so the expected lowering is present. Whispered is now
quieter than neutral: energy is 0.0398 versus 0.0513, and the intended quieting
is measurable. Both directions were previously masked by the demo writer's
peak normalization; the current writer applies only true-peak safety, so these
comparisons are now valid. Whispered is still a prosody/phonation
approximation, not a true unvoiced whisper. Accent F0 differences are
descriptive only; no accent target contour or accent-aware G2P exists yet.
Every condition has one intensity value, so intensity monotonicity is
`insufficient_data` rather than passed.

The older 29-asset synthesis evaluation (`results/synthesis_evaluation_v2.json`)
reported a 0.3260 overall WER and a 0.3529 accent-subset mean with Irish at
0.9412. That claim was real for the older WAVs, but it is superseded: the
current identity-controlled accent assets measure 0.1961 mean and 0.3529
Indian English. The older 0.353 figure was the accent subset, not the full
29-asset mean.

`scripts/audit_expression_data.py` writes
`data/processed/expression_data_audit.json`. It audits 13 datasets. VoxCeleb2 is
present and research/non-commercial. CREMA-D and the selected GLOBE_V3 audio are
now local. Appropriately licensed with obligations: CREMA-D, VCTK, OpenSLR 83,
Svarah, EdAcc, GLOBE_V3, and Common Voice English. MSP-Podcast requires an
institutional agreement; RAVDESS requires a commercial license; ESD, IEMOCAP,
L2-ARCTIC, and VoxCeleb2 are research-only. The recommended acquisition order is
CREMA-D first for pilot/evaluation emotion work, MSP-Podcast after the agreement
for learned emotion control, and GLOBE_V3 as the primary accent corpus.

CREMA-D curation (`scripts/curate_crema_d.py`) parsed all **7,442 clips** and
kept **6,892 clips / 91 actors / 4.869 h** after excluding 550 source-clipped
clips (7.4%). The strict clipping rule is the same one applied to GLOBE; nothing
was rescaled. The six intended classes measure **0.633–0.963 h** each (ratio
1.521) with 90–91 speakers and 9/9 val/test speakers, so every class fails the
2.0 h gate. CREMA-D is pilot/evaluation-only and cannot train a learned emotion
control. The voice-only crowd vote is joined as a perceptual audit label:
neutral is recognized at 0.976, but happy 0.330, sad 0.236, fear 0.356, disgust
0.343, and angry 0.568, which is why acted labels are not treated as clean
ground truth. The pilot classifier (`scripts/train_emotion_classifier.py`)
uses 125-D acoustic features and an RBF SVM selected on validation; on 699
held-out clips from 9 speakers it measures **57.8% accuracy / 0.578 macro-F1**
against a 17.9% majority baseline, with speaker-cluster bootstrap 95% CIs of
**0.513–0.647** and **0.511–0.643**. It is explicitly
`control_training_eligible: false`.

GLOBE_V3 has 723,532 rows (704,750 train / 9,603 dev / 9,179 test), but the
paper figures of 535 h / 23,519 speakers / 164 accents must not be reported as
V3 measurements; recompute them from V3 metadata. The Hugging Face tree is
**348.7 GB across 603 files**: 165.5 GB of `data/*.parquet` plus 183.2 GB of
duplicate `train/`/`dev`/`test` audio. The card's 165.5 GB download size is the
Parquet payload and 184.5 GB is the uncompressed dataset size, so the earlier
"stale" characterization was wrong. The `D:` drive has **138.1 GB free**, so a
full clone does not fit and is unnecessary. GLOBE_V3 has no `speaker_id`; join
`common_voice_path` to Common Voice 21 metadata to recover `client_id` before
constructing speaker-disjoint splits.

The metadata-first pipeline is implemented and measured. A full pass over 187
Parquet shards read only each footer and the ~20 KB metadata span per row
group: **723,532 rows, 0 failures, 858 s**. The Common Voice 21 join matched
**723,532/723,532 rows (100%)**, recovering **50,951 speakers / 1,102.1 h**;
450,835 rows have a self-reported accent. The strict selector produced **9,688
clips / 13.83 h pre-trim / 889 speakers / 543 row groups**. Five classes pass
at 2.50 h pre-trim each (ratio 1.0, 94–201 train speakers, ≥5 val/test
speakers); Irish is evaluation-only at 1.32 h. The selection uses 2.5 h
pre-trim because GLOBE audio is VAD-trimmed (measured mean −0.49 s per clip
across the full download, max 6.33 s on long-silence clips); the post-download
audit measures the authoritative post-trim hours and does not lower the 2.0 h
gate. The audit excluded 154 source-clipped clips (1.6%), leaving **9,534
clips / 12.28 h actual**. The five trainable classes measure 2.185–2.257 h
with 94–198 train speakers; Irish measures 1.176 h and remains
evaluation-only.

Access actions: CREMA-D is a direct GitHub download under ODbL 1.0 with no form
(attribute and cite it); MSP-Podcast requires the institution to sign the
academic license and email Prof. Carlos Busso; GLOBE_V3 is CC0 on Hugging Face;
Common Voice 21 metadata is a separate download required for the speaker join.
Human MOS/CMOS requires real listeners, informed consent, and likely IRB/ethics
review; no ratings exist yet.

These results do not support a production-grade expression claim. CREMA-D is
below the 2 h/class gate, so the learned emotion path is still blocked on
MSP-Podcast or another licensed corpus; the accent path is blocked on the
classifier/evaluation and accent-aware G2P; human ratings do not exist; and the
current accent path is regex respelling plus fixed deltas rather than
phonological control.

### 10.9 Residual noise and polished delivery: measured causes (2026-11-05 update)

**Current finding.** The v1 demo's persistent hiss came from the raw
face-derived style prior, not from clipping, the output writer, or the diffusion
step count. A controlled matrix held face, speaker embedding, text, and seed
fixed. Raw face style measured **-49.17 dBFS**; the same model with a real AMI
style measured **-71.07 dBFS**. Steps 5/20/30 and the N1 LoRA did not remove the
hiss (N1 measured -46.31 dBFS, worse than the raw baseline). The v2 demo applies
a gender-matched mean translation,
`s' = s + alpha * (mean_real_gender - mean_face_gender)`, at `alpha=1.0`.
On the three selected voices it cuts noise by **13.36, 17.58, and 17.12 dB**
(Lupita, Timothee, Jackie). The 21-clip v2 median is **-63.45 dBFS** (min
-64.78, max -61.75), with median SNR proxy **44.97 dB** (min 42.74, max 46.40),
versus a v1 12-clip median of **-49.17 dBFS**. The calibration artifact is
`results/audit/style_calibration/calibration.json`.

The human AMI reference median is **-76.11 dBFS** (8 clips), so v2 remains
**12.67 dB above** that reference. Alpha 1.25 was rejected: the selected-trio
median improves by only 0.04 dB (-63.51 vs -63.47 dBFS) while the
Timothee-Jackie ECAPA similarity rises to 0.422. This is an exploratory demo
transform, not a production noise specification, and no human MOS/CMOS or
noise-preference study exists.

**Historical diagnostic (one face, raw face style).** The earlier controlled
diagnostic is `scripts/diagnose_audio_quality.py`; it holds face, speaker
embedding, text, and seed fixed, disables watermarking to match the demo, and
measures the raw backend waveform and the waveform written by the current
`save_audio_sample()`. It established the writer and step-count behaviour but
did not isolate style as the hiss source; the later matrix did.

| Condition | Steps | Raw noise floor (dBFS) | Raw SNR proxy (dB) | Raw peak | Saved LUFS | Saved noise floor (dBFS) |
|---|---:|---:|---:|---:|---:|---:|
| neutral | 5 | -45.70 | 27.42 | 0.642 | -23.00 | -45.70 |
| whispered | 5 | -43.99 | 22.14 | 0.496 | -25.73 | -43.99 |
| neutral | 10 | -47.43 | 29.00 | 0.625 | -23.00 | -47.43 |
| whispered | 10 | -43.26 | 21.73 | 0.423 | -25.73 | -43.26 |
| neutral | 20 | -46.89 | 28.51 | 0.582 | -23.00 | -46.89 |
| whispered | 20 | -40.58 | 18.88 | 0.439 | -25.73 | -40.58 |
| neutral DSP control | 5 | -48.09 | 29.53 | 0.654 | -23.00 | -48.09 |
| neutral + pitch 1.16/time 0.92 | 5 | -42.92 | 23.91 | 0.531 | -23.00 | -42.92 |

**No clipping.** The clipping fraction is 0.0 in every condition. The highest
raw peak in the diagnostic is 0.654; the earlier demo's loudest saved file was
0.732. HF energy above 8 kHz is 4e-05–0.00097 of total power and spectral
flatness is 0.0034–0.0141. The residual is therefore not clipping or a broken
output stage;
the later controlled matrix showed that the raw face-style prior adds audible
broadband hiss, which the v2 calibration reduces.

**Double normalization fixed.** The backend normalizes to -23 LUFS and now
applies only a -1 dBTP true-peak limiter; `save_audio_sample()` no longer
peak-normalizes. The controlled A/B shows the legacy writer added +2.94 dB to
neutral and +5.17 dB to whispered at 5 steps (up to +6.55 dB across the sweep),
moving saved loudness to -20.6…-18.4 LUFS. The current writer adds 0.00 dB.
Neutral demo files are saved at -23.00 LUFS; expressive offsets are intentional
(happy -22.11, sad -24.26, angry -21.70, whispered -25.73, surprised -22.25).
This is why whispered is now quieter than neutral in the expression table.

**In that diagnostic, DSP dominated the step-count effect.** The neutral DSP
control is -48.09 dBFS / 29.53 dB. Applying only pitch 1.16 and time 0.92, with
the same style and step count, gives -42.92 dBFS / 23.91 dB. In that diagnostic
the emotion/accent path applied this global librosa pitch/time stretch, making
it a larger noise source than the number of diffusion steps; the later
controlled matrix showed the raw style prior is the larger source in the demo.

**Step count is not a monotonic quality knob here.** The single-seed run put
neutral at 10 steps first (-47.43 dBFS / 29.00 dB) and 20 slightly worse
(-46.89 / 28.51); whispered fell from 22.14 dB at 5 steps to 21.73 at 10 and
18.88 at 20. The follow-up 3-seed × 2-prompt ablation
(`results/step_ablation.json`, 48 rows) confirms the shape: neutral SNR is
27.06 ± 1.16 dB at 5, 27.71 ± 1.62 at 10, and 26.99 ± 2.13 at 20; whispered is
21.81 ± 0.76, 21.79 ± 0.61, and 19.81 ± 1.47. 10 steps beat 5 in 5/6 paired
samples (mean +0.65 dB, range -0.52…+1.58), whispered was a wash, and 20 was
2.0 dB worse for whispered. The tier-1 default is now 10. The result still
does not establish perceptual naturalness, and the other hardware tiers remain
unvalidated.

**Polished delivery (unchanged in v2).** The backend logs "Invalid or missing
model checkpoint path. Loading default model..." and loads StyleTTS2's default
LibriTTS checkpoint, an audiobook-narration model. The v2 demo renders one
formal 12-word prompt in about 7.1–8.1 s (~1.5–1.7 words/s) with no
disfluencies. The result is a careful narrator, not conversational speech.
This is a checkpoint/prompt/style mismatch; no post-processing step creates
it. The N1 LoRA did not fix it, and the controlled matrix found worse noise
under N1, so N1 is omitted from v2.

**Hardware-profile bug and demo versions.** `VoixPipeline.__init__` called
`detect_hardware()` without `configs/hardware.yaml`. That function uses the
fallback defaults when no path is supplied, and the fallback is 5 diffusion
steps for every tier; the YAML profile for the GTX 1650 is 20. The pipeline and
`VoixService.get_diagnostics()` now use `get_hardware_profile()`, which loads
the YAML profile. Regression tests cover both the YAML path and the constructed
pipeline. The v2 research demo was generated with an explicit
`--diffusion-steps 10`, seed 42, and the calibrated styles; its metadata records
10 steps and the checked-in audio is reproducible. The multi-seed ablation
still supports 10 as the tier-1 default. The older 15-asset demo remains at
`results/demo/research/` as a fixed v1 artifact and is not the current research
demo.

The pre-phase-0 snapshot was not reproducible from the current checkpoints
despite matching seed/config: the regenerated waveforms differ materially, not
just in gain. It was moved to
`D:\voix_archive\20261105_cleanup\files\results\demo_pre_phase0` during the
approved 2026-11-05 cleanup. The v2 research demo
(`results/demo/research_v2/`) is the current canonical artifact; all v2
measurements are in its `report.json`.

**What this does not establish.** The historical table is a diagnostic proxy
from one face, one prompt, one seed, and one clip per condition. The v2 matrix
adds three faces and 21 clips, but still has no human ratings. Neither
establishes MOS, naturalness, or a production noise specification; a proper
step-count and DSP ablation needs multiple prompts and seeds, confidence
intervals, and human listening.

### 10.10 StyleTTS2 prosody LoRA pilot (research only)

**Question.** The polished, audiobook-like delivery comes from the frozen
LibriTTS StyleTTS2 checkpoint plus a formal prompt. Can a small LoRA adapter on
the prosody diffusion branch move the model toward conversational speech
without touching the identity path?

**Corpus.** `scripts/build_styletts2_lora_corpus.py` selected a deterministic,
gender-balanced, speaker-disjoint VoxCeleb2 subset and extracted 24 kHz audio.
Whisper `base` produced real ASR pseudo-transcripts with quality filtering.
Measured corpus: **432 clips / 263 speakers / 0.741 h**; train **369 clips /
225 speakers / 0.628 h**, val **63 clips / 38 speakers / 0.113 h**. Of 700
candidate targets, **268 were dropped**: 202 low Whisper log-probability, 47 too
long, 16 too few words, and 3 text too long. The transcripts are ASR output,
not human ground truth, and VoxCeleb2 is research/non-commercial. This is a
pilot, not a naturalness benchmark.

**Method.** `scripts/train_styletts2_lora.py` froze every StyleTTS2 component,
including the CFVN/CVAE/style-adapter path, and injected manual rank-8 /
alpha-16 LoRA into 25 `nn.Linear` layers of `model.diffusion.unet`: **434,184
trainable parameters**. The adapter is zero-initialised, so it starts exactly
equal to the base model. Training used AdamW at `2e-4`, weight decay `0.01`,
gradient accumulation 4, warmup-cosine, 10 epochs, patience 4, seed 42, and
three validation repeats. The run completed all 10 epochs in 488.9 s with
**0 failed clips**. Train loss was 0.5469; validation diffusion loss fell from
0.5980 (epoch 1) to **0.5226 (epoch 10, best)**. The final `lora_B` norms were
0.424–0.866 (mean 0.638), so the adapter genuinely moved off zero.

**Paired generation evaluation.** The best checkpoint was compared with the
base model on 25 held-out clips from 21 speakers. Both conditions used the same
text, reference style, and random seed; generation used 10 diffusion steps,
`alpha=0.18`, `beta=0.22`, and the normal `-23 LUFS` output stage. Whisper
`base` measured WER. ECAPA measured similarity to the target clip and to the
different-video reference clip. The table reports paired within-clip
differences and 95% paired t confidence intervals; p-values are uncorrected
except where the Holm result is noted below. Raw rows are in
`results/checkpoints/styletts2_lora_prosody_generation_eval.json`; training
history is in `results/checkpoints/styletts2_lora_prosody.history.json`.

| Metric | Base (mean ± SD) | LoRA (mean ± SD) | Paired delta (LoRA − base) | 95% CI | p |
|---|---:|---:|---:|---:|---:|
| WER | 0.0331 ± 0.0443 | 0.0354 ± 0.0475 | +0.0022 | −0.0122…+0.0166 | 0.754 |
| ECAPA target similarity | 0.1222 ± 0.0890 | 0.1267 ± 0.0952 | +0.0045 | −0.0085…+0.0174 | 0.483 |
| ECAPA reference similarity | 0.2400 ± 0.1324 | 0.2362 ± 0.1272 | −0.0039 | −0.0160…+0.0082 | 0.514 |
| F0 mean (Hz) | 162.76 ± 46.62 | 157.95 ± 47.59 | −4.80 | −9.62…+0.01 | 0.051 |
| F0 SD (Hz) | 29.54 ± 21.68 | 29.29 ± 16.84 | −0.25 | −5.43…+4.93 | 0.921 |
| F0 range (Hz) | 138.50 ± 87.37 | 137.67 ± 93.69 | −0.83 | −20.57…+18.91 | 0.932 |
| Voiced ratio | 0.5329 ± 0.2000 | 0.5182 ± 0.1984 | −0.0146 | −0.0321…+0.0028 | 0.096 |
| Duration (s) | 7.303 ± 2.158 | 7.289 ± 2.162 | −0.014 | −0.046…+0.018 | 0.377 |
| Energy mean | 0.05336 ± 0.00466 | 0.05377 ± 0.00435 | +0.00041 | −0.00029…+0.00110 | 0.240 |
| Energy SD | 0.04572 ± 0.00430 | 0.04457 ± 0.00423 | −0.00115 | −0.00182…−0.00049 | 0.0015 |

After Holm correction across the ten metrics, only the energy-SD decrease
remains significant (adjusted p = 0.015). That is a small signal-statistics
change, not evidence of better naturalness, and it does not survive as a
perceptual claim. WER, speaker similarity, F0, duration, and energy mean all
have confidence intervals that include zero.

**Interpretation and decision.** The adapter learned the diffusion objective
but did not improve the objective naturalness proxies. The correct conclusion
is a **null result on the objective proxies**, not a quality win. Human
MOS/CMOS is unavailable, so naturalness itself remains unmeasured. The
adapter is experimental and is **not wired into `StyleTTS2Backend`**; the
canonical CFVN (`results/checkpoints/cfvn_v2_best.pt`) and style adapter
(`checkpoints/style_adapter_cfvn_best.pt`) are unchanged. The next credible
step is a confirmatory conversational run with more held-out speakers, multiple
seeds, the full pre-registered predictor+diffusion adapter, and real listener
ratings; neither the 0.741 h VoxCeleb2 corpus nor the four-speaker AMI N1 pilot
is large enough for a production naturalness claim.

**Later demo-matrix decision (2026-11-05).** A separate controlled matrix held
the face, speaker embedding, text, and seed fixed and compared the N1 LoRA with
the raw face-style baseline at demo settings. N1 measured **-46.31 dBFS /
28.40 dB SNR**, versus **-49.17 dBFS / 31.28 dB** for the raw baseline: 2.86 dB
worse noise, and about 17 dB worse than the calibrated v2 median
(-63.45 dBFS). The N1 condition is therefore omitted from the v2 demo. This
does not overturn the narrow F0-proxy gate result in
`results/naturalness/FINDINGS.md`; it adds a deployment-relevant noise cost that
the gate did not test. The N1 checkpoint remains a research artifact and is not
wired into serving.
