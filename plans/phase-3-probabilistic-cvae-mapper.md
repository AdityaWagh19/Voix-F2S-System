# Phase 3: Probabilistic CVAE & Metric Contrastive Mapper

> **Phase Identifier:** `PHASE-3`  
> **Target Duration:** Weeks 4–6  
> **Status:** Complete (Executed & Verified)  
> **Prerequisites:** `PHASE-2` Complete  

---

## 1. Objective

Implement, optimize, and validate the mapping from unified facial representations $e_f \in \mathbb{R}^{560}$ to speaker identity embedding distributions in ECAPA-TDNN acoustic space ($\mathbb{R}^{192}$). Implement both the Conditional Variational Autoencoder (CVAE) probabilistic mapper and the Metric Contrastive Mapper. Directly solve the fundamental "acoustic mean collapse" problem discovered in unconstrained regression by employing Supervised InfoNCE loss over speaker identity centroids with auxiliary demographic supervision.

---

## 2. Scope

### In Scope
- Recognition encoder $q_\phi(z \mid e_s, e_f)$, conditional prior $p_\theta(z \mid e_f)$, and generative decoder $p_\psi(e_s \mid z, e_f)$ operating in latent space $\mathbb{R}^{192}$.
- Differentiable reparameterization trick with numerical clamping safeguards.
- Directional reconstruction loss and dimension-wise free-bits KL divergence ($\lambda_{\text{fb}} = 0.5$ nats/dim) with linear $\beta(t)$ annealing.
- Parametric **Metric Contrastive Mapper** (`MetricContrastiveMapper` in `voix/models/cvae.py`):
  - 4-layer MLP backbone with BatchNorm1d, GELU, and Dropout.
  - L2 normalization head projecting directly onto unit hypersphere $\mathbb{S}^{191}$.
  - Auxiliary demographic gender classification head (`Linear(256, 1) -> Sigmoid`).
- Supervised InfoNCE contrastive loss over speaker centroids with temperature $\tau = 0.07$.
- Mixed precision (FP16/AMP) training loop, AdamW optimizer, and cosine learning rate scheduler.
- Checkpointing and validation against 400 strictly unseen zero-shot validation identities.

### Out of Scope
- Direct raw audio waveform generation (handled downstream by frozen StyleTTS2 in Phase 4).
- End-to-end backpropagation through the face or speech backbones (InsightFace and SpeechBrain remain strictly frozen).
- Dynamic facial expression or motion tracking.

---

## 3. Design Decisions & Rationale

### 3.1 The Acoustic Mean Collapse Phenomenon
In early experiments, training standard CVAEs or direct regression networks with MSE / Cosine distance directly on single-utterance acoustic targets caused severe **acoustic collapse**:
- Single speakers exhibit substantial acoustic variance across utterances due to prosody, room acoustics, and emotion.
- Point-wise regression losses on single utterances force the network toward the global mean of the entire training corpus.
- Samples drawn from the population mean sound robotic, muffled, and lack distinct vocal identity.

### 3.2 Supervised InfoNCE Loss over Speaker Centroids
To eliminate single-utterance noise and prevent mean collapse, VOIX introduces clean speaker centroid targets:
$$\mathbf{c}_y = \frac{1}{|U_y|} \sum_{u \in U_y} \mathbf{e}_{s, u}, \quad \mathbf{c}_y \leftarrow \frac{\mathbf{c}_y}{\|\mathbf{c}_y\|_2}$$
The mapper is trained using Supervised InfoNCE:
$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\hat{\mathbf{e}}_s \cdot \mathbf{c}_{y} / \tau)}{\sum_{j=1}^{C} \exp(\hat{\mathbf{e}}_s \cdot \mathbf{c}_j / \tau)}$$
- **Temperature $\tau = 0.07$:** Sharpens the contrastive distribution, strongly penalizing predictions that collapse toward the center of the sphere.
- **Centroids:** Ensures the target is pure speaker identity rather than transient utterance acoustics.

### 3.3 Auxiliary Demographic Gender Head
Voice pitch ($F_0$) and vocal tract scale differ substantially across biological sex. Adding an auxiliary classification loss:
$$\mathcal{L}_{\text{gender}} = \text{BCE}(\hat{y}_{\text{gender}}, y_{\text{gender}})$$
forces internal layers to form linearly separable representations, preventing cross-gender vocal artifacts and boosting gender consistency to 100% when coupled with retrieval.

---

## 4. Sequential Implementation Tasks

### T3.1: Probabilistic CVAE Implementation
- [x] Implement $q_\phi(z \mid e_s, e_f)$, $p_\theta(z \mid e_f)$, and $p_\psi(e_s \mid z, e_f)$ in `voix/models/cvae.py`.
- [x] Reparameterization with numerical log-variance clamping: `torch.clamp(logvar, min=-10.0, max=10.0)`.
- [x] Free-bits KL loss implementation with $\lambda_{\text{fb}} = 0.5$ nats/dim.

### T3.2: Metric Contrastive Mapper Implementation
- [x] Implement `MetricContrastiveMapper` (`560 -> 512 -> 384 -> 256 -> 192` + auxiliary head).
- [x] Enforce unit hypersphere normalization: `F.normalize(x, p=2, dim=-1)`.
- [x] Verify gradient propagation across both heads.

### T3.3: Supervised InfoNCE Loss & Batching
- [x] Implement `SupervisedInfoNCELoss` with temperature $\tau = 0.07$.
- [x] Implement identity batch sampler ensuring multi-positive and negative identity balance per batch.

### T3.4: Training Execution & Hyperparameter Tuning
- [x] Train 40 epochs on GPU using AdamW ($lr=1\times 10^{-3}$, weight decay $1\times 10^{-4}$), Cosine Annealing, AMP FP16.
- [x] Validate model convergence on validation set.

### T3.5: Validation on 400 Zero-Shot Held-Out Speakers
- [x] Evaluate checkpoint on 10,000 utterances across 400 strictly unseen identities.
- [x] Checkpoint `D_joint_infonce_best.pt` achieved validation cosine similarity of **0.1288** and peak speaker SECS of **0.1877**.

---

## 5. Acceptance Criteria & Test Matrix

| Test Suite / Metric | Target Criteria | Actual Verified Result | Status |
|---|---|---|---|
| Latent Dimension Invariance | Input `[B, 560]` $\to$ Output `[B, 192]` unit-norm | Exact shape `[B, 192]`, norm $1.0 \pm 10^{-5}$ | Passed |
| Numerical Stability | No NaNs/Infs under FP16 training | 0 NaNs across 40 epochs | Passed |
| Zero-Shot Validation Cosine | $> 0.100$ on unseen validation speakers | **0.1288** | Passed |
| Auxiliary Gender Accuracy | $> 60\%$ on zero-shot validation faces | **62.6%** (Parametric), **100.0%** (Hybrid) | Passed |
| Unit Test Suite | `pytest tests/test_cvae.py` passes all fixtures | 100% Passed | Passed |

---

## 6. Deliverables & Production Artifacts

1. `voix/models/cvae.py`: `ProbabilisticCVAEMapper`, `MetricContrastiveMapper`, and loss functions.
2. `results/checkpoints/D_joint_infonce_best.pt`: Best metric contrastive checkpoint trained with Supervised InfoNCE over centroids.
3. `data/processed/retrieval_gallery.pt`: 3,029 precomputed speaker centroids for downstream hybrid inference.

---

*Read next: [phase-4-acoustic-synthesis-and-style-adapter.md](phase-4-acoustic-synthesis-and-style-adapter.md)*
