# Phase 6: Baselines, Ablations & Empirical Evaluation

> **Phase Identifier:** `PHASE-6`  
> **Target Duration:** Weeks 12–14  
> **Status:** Pending Execution  
> **Prerequisites:** `PHASE-3`, `PHASE-4`, and `PHASE-5` Complete

---

## 1. Objective

Conduct an exhaustive empirical evaluation of the VOIX research system. Implement and benchmark against six baseline models (B1–B6). Execute the complete feature ablation matrix (A1–A6) and architectural ablation matrix (B1–B4). Quantify cross-demographic domain degradation and recovery on the India F2S Benchmark (evaluating RQ3/H3). Compute SHAP feature attributions on craniofacial ratios to evaluate biomechanical hypotheses (RQ2/H2). Execute objective (SECS, Recall@K, DS, ECE, FSD) and subjective (MOS, A/B preference, diversity preference) evaluation protocols with statistical significance testing.

---

## 2. Scope

### In Scope
- Implementation of 6 baseline architectures:
  - **B1 (Uniform Random):** Random unit-norm vector in ECAPA-TDNN space.
  - **B2 (Nearest Neighbor):** 1-NN cosine face retrieval against training identity dictionary.
  - **B3 (Deterministic Regression):** 4-layer MLP mapping $e_f \to \hat{e}_s$ without stochastic sampling ($\beta = 0$).
  - **B4 (ArcFace-Only CVAE):** Variational autoencoder conditioned purely on 512-D biometric identity.
  - **B5 (Zero-Shot F2S):** Direct re-implementation of state-of-the-art deterministic cross-modal baseline.
  - **B6 (GMM Mapper):** Conditional Gaussian Mixture Model baseline.
- Execution of Feature Ablations:
  - A1: ArcFace (512-D)
  - A2: FaceMesh ratios (32-D)
  - A3: Soft demographics (16-D)
  - A4: ArcFace + FaceMesh (544-D)
  - A5: ArcFace + Demographics (528-D)
  - A6: Full Fusion (560-D)
- Execution of Architecture Ablations:
  - B1: $\beta = 0$ throughout (Deterministic)
  - B2: CVAE without free bits ($\lambda_{\text{fb}} = 0$)
  - B3: CVAE without linear $\beta$ annealing schedule
  - B4: Proposed CVAE (Free bits + Cyclical annealing)
- Objective evaluation metrics: SECS, Recall@1, Recall@5, Recall@10, Diversity Score (DS), Expected Calibration Error (ECE), and Frechet Speaker Distance (FSD).
- Domain adaptation experiment: fine-tuning on India F2S Benchmark (5–10 epochs) and evaluating catastrophic forgetting on VoxCeleb1.
- SHAP feature attribution over all 32 craniofacial geometric dimensions against fundamental frequency ($F_0$), formant frequencies ($F_1, F_2$), and spectral centroid.
- Subjective evaluation toolkit: ITU-T P.808 MOS generation, A/B preference test pairing, diversity evaluation, and Krippendorff's alpha inter-rater reliability.
- Statistical significance tests: two-tailed paired t-test and Wilcoxon signed-rank test.

### Out of Scope
- Full retraining of the frozen StyleTTS 2 acoustic synthesis backend.
- Commercial crowdsourced MOS deployment logistics (code generation and trial preparation are in scope; payment execution handled by researcher).

---

## 3. Design Decisions & Rationale

### 3.1 Paired Metrics: Joint SECS and Diversity Score (DS)
- **Decision:** Never report Diversity Score in isolation. All tables must report DS alongside SECS.
- **Rationale:** A model generating pure uniform white noise produces a maximal Diversity Score ($DS \approx 1.0$), but has zero speaker consistency ($SECS \approx 0.0$). A scientifically valid probabilistic face-to-voice model must demonstrate high SECS *and* calibrated DS simultaneously.

### 3.2 Closed-Set Identity Retrieval (Recall@K) Protocol
- **Decision:** Evaluate Recall@K on a closed candidate pool of $N = 100$ identities from VoxCeleb1 test set, computing cosine similarity between predicted $\hat{e}_s$ and ground-truth speaker embeddings.
- **Rationale:** Recall@K tests cross-modal biometric alignment directly in embedding space, bypassing any vocoder artifacts introduced by the TTS backend.

### 3.3 Catastrophic Forgetting Safeguard
- **Decision:** After fine-tuning on the Indian benchmark, the model must be re-evaluated on VoxCeleb1. Performance retention must remain within 5% of the pre-fine-tuned baseline.
- **Rationale:** Domain adaptation is scientifically valid only if the model adapts to new acoustic and morphological patterns without destroying its learned representations on general populations.

---

## 4. Sequential Implementation Tasks

```
[T6.1] Baseline Suite Implementation (B1 to B6)
       ├── Implement RandomBaseline (B1) and NNRetrievalBaseline (B2)
       ├── Implement DeterministicMLP (B3)
       ├── Implement ArcFaceOnlyCVAE (B4)
       └── Implement ZeroShotF2S (B5) and GMMBaseline (B6)

[T6.2] Comprehensive Evaluation Metrics Engine
       ├── Implement SECS, Recall@K (K=1, 5, 10), and DiversityScore
       ├── Implement ExpectedCalibrationError (ECE)
       └── Implement FrechetSpeakerDistance (FSD)

[T6.3] Feature Ablation Experiment Runner (A1 to A6)
       ├── Automate training across feature subsets
       └── Generate comparative metrics table

[T6.4] Architecture Ablation Experiment Runner (B1 to B4)
       ├── Automate training across loss/annealing configurations
       └── Measure active latent units and posterior collapse rates

[T6.5] Cross-Domain Evaluation & Demographic Gap Analysis
       ├── Evaluate VoxCeleb-trained model zero-shot on India F2S Benchmark
       └── Verify Hypothesis H3 (SECS drop >= 10%)

[T6.6] Domain Adaptation Fine-Tuning & Retention Audit
       ├── Fine-tune on India F2S Benchmark (5-10 epochs)
       └── Evaluate on India Benchmark and VoxCeleb1 (catastrophic forgetting audit)

[T6.7] SHAP Feature Attribution & Biomechanical Correlation
       ├── Compute KernelSHAP on 32-D FaceMesh ratios
       └── Correlate top ratios with acoustic F0 and formant shifts

[T6.8] Subjective Evaluation Toolkit & Statistical Testing
       ├── Script ITU-T P.808 sample selector (50 audio pairs)
       ├── Implement Wilcoxon signed-rank and paired t-test analyzers
       └── Export publication-ready LaTeX tables
```

### Detailed Task Specifications

#### Task 6.1: Baseline Suite
- File: `voix/evaluation/baselines.py`
- `DeterministicMLP` (B3): Identical layers to CVAE, but predicting $\hat{e}_s$ directly with $\beta = 0$.
- `NNRetrievalBaseline` (B2): Stores training ArcFace bank; retrieves $\arg\max_i \cos(e_{\text{face}}, e_{\text{face}}^{(i)})$ and returns corresponding $e_s^{(i)}$.

#### Task 6.2: Metrics Implementation
- File: `voix/evaluation/metrics.py`
  ```python
  def compute_secs(e_s_hat_samples: torch.Tensor, e_s_gt: torch.Tensor) -> float:
      # e_s_hat_samples: (B, K, 192), e_s_gt: (B, 192)
      sims = F.cosine_similarity(e_s_hat_samples, e_s_gt.unsqueeze(1), dim=-1)
      return sims.mean().item()

  def compute_diversity_score(e_s_hat_samples: torch.Tensor) -> float:
      # e_s_hat_samples: (B, K, 192)
      # Computes mean pairwise cosine distance across K samples
      B, K, D = e_s_hat_samples.shape
      dist_sum = 0.0
      count = 0
      for i in range(K):
          for j in range(i + 1, K):
              dist_sum += (1.0 - F.cosine_similarity(e_s_hat_samples[:, i], e_s_hat_samples[:, j], dim=-1)).sum()
              count += B
      return (dist_sum / count).item()
  ```

#### Task 6.5 & 6.6: Demographic Gap & Fine-Tuning
- File: `scripts/run_cross_domain_eval.py`
- Step 1: Zero-shot test on `India F2S Benchmark`: record $\text{SECS}_{\text{zero-shot}}$.
- Step 2: Compare with $\text{SECS}_{\text{voxceleb1}}$:
  $$\Delta_{\text{demog}} = \frac{\text{SECS}_{\text{voxceleb1}} - \text{SECS}_{\text{zero-shot}}}{\text{SECS}_{\text{voxceleb1}}} \times 100\%$$
- Step 3: Fine-tune CVAE prior and decoder for 5 epochs on Indian dataset (`lr=5e-5`).
- Step 4: Re-evaluate on both datasets. Verify $\Delta_{\text{voxceleb1}} \le 5.0\%$.

#### Task 6.7: SHAP Feature Attribution
- File: `voix/evaluation/shap_analysis.py`
- Target outputs:
  - Acoustic pitch ($F_0$ mean extracted via `praat-parselmouth` or `pyin`).
  - First two formant frequencies ($F_1, F_2$).
- Compute SHAP values for each of the 32 FaceMesh ratio dimensions.
- Identify the top 5 predictive geometric features and plot summary beeswarm charts to `artifacts/shap_ratios.png`.

---

## 5. Validation Strategy

### Automated Metric Suite
```bash
python scripts/run_evaluation.py --checkpoint checkpoints/cvae_best.pt --test-set data/processed/voxceleb1_test.h5
```

### Ablation Matrix Execution
```bash
python scripts/run_ablations.py --config configs/evaluation.yaml
```

### Statistical Significance Verification
```bash
python scripts/compute_statistics.py --results-csv results/ablation_results.csv
```

---

## 6. Acceptance Criteria

1. **Hypothesis 1 (Probabilistic Modeling):**
   - Full CVAE model achieves $\text{SECS} \ge 0.68$ on VoxCeleb1 while maintaining calibrated Diversity Score $DS \in [0.18, 0.35]$.
   - CVAE statistically outperforms Deterministic Baseline B3 in subjective A/B preference tests ($p < 0.01$).
2. **Hypothesis 2 (Feature Disentanglement):**
   - Full model A6 ($\text{ArcFace} + \text{Morphology} + \text{Demographics}$) statistically outperforms ArcFace-only A1 ($p < 0.05$ via paired t-test on SECS).
   - SHAP feature analysis identifies at least 3 craniofacial ratios with statistically significant predictive attribution ($p < 0.01$) regarding $F_0$ and formant structure.
3. **Hypothesis 3 (Demographic Generalization):**
   - Western-trained model exhibits a statistically significant performance drop ($\ge 10.0\%$ SECS decrease) when evaluated zero-shot on the India F2S Benchmark.
   - Domain adaptation fine-tuning recovers at least $60\%$ of this gap while retaining $\ge 95\%$ of original performance on VoxCeleb1 (no catastrophic forgetting).
4. **Baseline Superiority:** Proposed system outperforms Baselines B1, B2, B3, and B4 on joint SECS and Recall@5 metrics.
5. **Statistical Rigor:** All reported metrics include mean and standard deviation over 3 fixed random seeds.

---

## 7. Risks & Trade-offs

| Risk | Impact | Mitigation Strategy |
|---|---|---|
| Hypothesis H2 fails (craniofacial ratios add no value) | Weakened novelty claim | Publish as a definitive negative empirical finding disproving common biomechanical assumptions in literature |
| Fine-tuning causes catastrophic forgetting on VoxCeleb1 | Model regression | Apply Elastic Weight Consolidation (EWC) or replay buffer with 20% VoxCeleb2 samples |
| Subjective study inter-rater agreement is low | Invalid MOS results | Filter evaluators failing attention trap questions; recompute with Krippendorff's $\alpha \ge 0.65$ |

---

## 8. Deliverables

- `voix/evaluation/baselines.py`: Complete B1–B6 baseline implementations.
- `voix/evaluation/metrics.py`: Standardized metric calculation suite.
- `voix/evaluation/shap_analysis.py`: SHAP attribution script.
- `scripts/run_ablations.py`: Automated ablation matrix harness.
- `scripts/run_cross_domain_eval.py`: Demographic evaluation script.
- `results/master_results_table.csv`: Comprehensive metric matrix.
- `artifacts/shap_ratios.png`: Biomechanical attribution visualization.

---

## 9. Documentation Updates

- Document full ablation results in `project-context/architecture.md` Section 9.
- Update `project-context/mvp.md` validation gate outcomes.

---

## 10. Dependencies

- **Predecessor:** `PHASE-3`, `PHASE-4`, and `PHASE-5`.
- **Successor:** `PHASE-7` (Production Hardening & Reproducibility).
