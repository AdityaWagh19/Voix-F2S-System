# Phase 3: Probabilistic CVAE Mapper

> **Phase Identifier:** `PHASE-3`  
> **Target Duration:** Weeks 4–6  
> **Status:** Pending Execution  
> **Prerequisites:** `PHASE-2` Complete

---

## 1. Objective

Implement, optimize, and validate the Conditional Variational Autoencoder (CVAE) probabilistic mapper that models the stochastic mapping from unified facial representations $e_f \in \mathbb{R}^{560}$ to speaker identity embedding distributions $P(e_s \mid e_f) = \mathcal{N}(\mu_\theta(e_f), \text{diag}(\sigma_\theta^2(e_f)))$ in ECAPA-TDNN acoustic space ($\mathbb{R}^{192}$). Ensure robust optimization without posterior collapse using free-bits KL regularization and linear beta annealing.

---

## 2. Scope

### In Scope
- CVAE recognition encoder $q_\phi(z \mid e_s, e_f)$ and prior/generation network $p_\theta(z \mid e_f)$ operating in latent space $\mathbb{R}^{192}$.
- Generative decoder $p_\theta(e_s \mid z, e_f)$ predicting reconstructed speaker embeddings $\hat{e}_s \in \mathbb{R}^{192}$.
- Differentiable reparameterization trick with gradient isolation safeguards.
- Directional reconstruction loss $\mathcal{L}_{\text{recon}} = 1 - \text{cosine\_similarity}(\hat{e}_s, e_s)$.
- Dimension-wise free-bits KL divergence loss ($\lambda_{\text{fb}} = 0.5$ nats/dim).
- Dynamic beta annealing scheduler $\beta(t)$ with 30% step linear warmup.
- Training loop featuring mixed precision (FP16/AMP), gradient clipping, checkpointing, and Weights & Biases telemetry.
- Hyperparameter grid search across $\beta_{\text{max}} \in \{0.1, 0.5, 1.0\}$ and learning rates $\{1\times 10^{-4}, 5\times 10^{-4}\}$.
- Posterior collapse and calibration diagnostic suite (active dimension count, mutual information proxy, Expected Calibration Error).

### Out of Scope
- Acoustic speech waveform generation via StyleTTS 2 (deferred to Phase 4).
- Fine-tuning on Indian benchmark data (deferred to Phase 6).
- Subjective perceptual listening tests (deferred to Phase 6).

---

## 3. Design Decisions & Rationale

### 3.1 Cosine Similarity Loss vs. Mean Squared Error (MSE)
- **Decision:** Optimize reconstruction in embedding space using cosine distance ($1 - \cos(\hat{e}_s, e_s)$) rather than Euclidean MSE ($\|\hat{e}_s - e_s\|_2^2$).
- **Rationale:** ECAPA-TDNN speaker embeddings reside on a hypersphere where identity discrimination is purely angular. Minimizing MSE penalizes magnitude deviations that have zero physical or acoustic significance, leading to blurred average predictions. Cosine loss directly aligns the directional vector.

### 3.2 Free-Bits KL Regularization ($\lambda_{\text{fb}} = 0.5$ nats)
- **Decision:** Implement free bits per latent dimension: $\mathcal{L}_{\text{KL}} = \sum_{d=1}^{192} \max(\lambda_{\text{fb}}, D_{\text{KL}}(q_\phi(z_d \mid e_s, e_f) \parallel p_\theta(z_d \mid e_f)))$.
- **Rationale:** Standard VAE training on continuous embeddings frequently suffers from catastrophic posterior collapse, where the encoder sets $\mu_d = 0, \sigma_d = 1$ to satisfy the KL penalty at the expense of ignoring facial conditioning. Free bits guarantees that the optimizer incurs zero penalty until each latent dimension encodes at least $0.5$ nats of information, forcing the network to utilize the latent capacity.

### 3.3 Conditional Architecture Formulation
- **Decision:** Condition both the encoder $q_\phi(z \mid e_s, e_f)$ and generative prior $p_\theta(z \mid e_f)$ on the face vector $e_f$. The decoder reconstructs $e_s$ from $z$ and $e_f$.
- **Rationale:** Standard VAEs condition only the decoder. Conditioning the prior network allows the latent space $z$ to capture *residual voice factors* (unobservable physiological variance, vocal habits) while the facial representation anchors the mean acoustic position.

---

## 4. Sequential Implementation Tasks

```
[T3.1] Neural Network Architecture Definition
       ├── Implement CVAEEncoder (Linear 752 -> 512 -> 384 -> 256 -> mu, logvar)
       ├── Implement CVAEPrior (Linear 560 -> 384 -> 256 -> mu, logvar)
       ├── Implement CVAEDecoder (Linear 752 -> 384 -> 256 -> 192)
       └── Verify tensor contracts and parameter initialization

[T3.2] Reparameterization Module & Sampling
       ├── Implement Reparameterize layer: z = mu + sigma * epsilon
       └── Implement test_gradient_flow verifying backpropagation

[T3.3] Loss Formulations & Free-Bits Implementation
       ├── Implement CosineReconstructionLoss
       ├── Implement FreeBitsKLLoss (lambda_fb thresholding)
       └── Unit test asserting zero loss gradient below threshold

[T3.4] Optimization Schedulers & Beta Annealing
       ├── Implement LinearWarmupScheduler for beta(t)
       └── Implement AdamW optimizer with cosine learning rate decay

[T3.5] Training Engine & Checkpointing
       ├── Implement TrainerCVAE class with PyTorch AMP
       ├── Implement automated validation loop & best-model checkpointing
       └── Connect Weights & Biases metric logging

[T3.6] Diagnostic & Evaluation Tools
       ├── Implement ActiveUnitsCounter (KL > 0.1 nats per dim)
       ├── Implement DiversityScoreEvaluator (pairwise cosine distance)
       └── Implement ExpectedCalibrationError (ECE) on latent confidence intervals

[T3.7] Hyperparameter Grid Search & Checkpoint Selection
       ├── Run grid across beta_max in {0.1, 0.5, 1.0} and lr in {1e-4, 5e-4}
       └── Select canonical checkpoint based on Pareto optimality (val SECS vs ECE)
```

### Detailed Task Specifications

#### Task 3.1 & 3.2: CVAE Architecture Implementation
- File: `voix/models/cvae.py`
- Architectural specifications:
  ```python
  class CVAE(nn.Module):
      def __init__(self, face_dim=560, speaker_dim=192, latent_dim=192):
          super().__init__()
          # Recognition Encoder: q_phi(z | e_s, e_f)
          self.enc_net = nn.Sequential(
              nn.Linear(face_dim + speaker_dim, 512),
              nn.BatchNorm1d(512),
              nn.GELU(),
              nn.Linear(512, 384),
              nn.BatchNorm1d(384),
              nn.GELU(),
              nn.Linear(384, 256),
              nn.BatchNorm1d(256),
              nn.GELU()
          )
          self.enc_mu = nn.Linear(256, latent_dim)
          self.enc_logvar = nn.Linear(256, latent_dim)

          # Prior Network: p_theta(z | e_f)
          self.prior_net = nn.Sequential(
              nn.Linear(face_dim, 384),
              nn.BatchNorm1d(384),
              nn.GELU(),
              nn.Linear(384, 256),
              nn.BatchNorm1d(256),
              nn.GELU()
          )
          self.prior_mu = nn.Linear(256, latent_dim)
          self.prior_logvar = nn.Linear(256, latent_dim)

          # Generative Decoder: p_theta(e_s | z, e_f)
          self.dec_net = nn.Sequential(
              nn.Linear(latent_dim + face_dim, 384),
              nn.BatchNorm1d(384),
              nn.GELU(),
              nn.Linear(384, 256),
              nn.BatchNorm1d(256),
              nn.GELU(),
              nn.Linear(256, speaker_dim)
          )

      def reparameterize(self, mu, logvar):
          std = torch.exp(0.5 * logvar)
          eps = torch.randn_like(std)
          return mu + eps * std
  ```

#### Task 3.3 & 3.4: Loss Functions & Schedulers
- File: `voix/training/losses.py`
- Formulate:
  ```python
  class CVAELoss(nn.Module):
      def __init__(self, lambda_fb=0.5):
          super().__init__()
          self.lambda_fb = lambda_fb

      def forward(self, e_s_hat, e_s, mu_q, logvar_q, mu_p, logvar_p, beta):
          # Reconstruction: Cosine distance
          recon_loss = 1.0 - F.cosine_similarity(e_s_hat, e_s, dim=-1).mean()

          # Analytical KL between two diagonal Gaussians
          # KL(q || p) = 0.5 * [log(sigma_p^2 / sigma_q^2) + (sigma_q^2 + (mu_q - mu_p)^2)/sigma_p^2 - 1]
          var_q = torch.exp(logvar_q)
          var_p = torch.exp(logvar_p)
          kl_dim = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p)**2) / (var_p + 1e-8) - 1.0)

          # Free-bits thresholding per dimension
          free_bits_kl = torch.clamp(kl_dim, min=self.lambda_fb).sum(dim=-1).mean()

          total_loss = recon_loss + beta * free_bits_kl
          return total_loss, recon_loss, free_bits_kl
  ```

#### Task 3.5 & 3.6: Training Loop & Diagnostics
- File: `voix/training/trainer_cvae.py`
- Configuration file: `configs/cvae_training.yaml`
  - Optimizer: `AdamW(lr=1e-4, weight_decay=1e-5)`
  - Batch size: 64 (fits within 2 GB VRAM on RTX 4050)
  - Epochs: 50
  - Beta warmup steps: $0.30 \times \text{total\_steps}$
- Diagnostic calculations logged every epoch:
  - `active_dimensions`: Count of dimensions where mean $\text{KL} > 0.1$ nats.
  - `val_cosine_sim`: Cosine similarity on held-out VoxCeleb1 evaluation set.
  - `diversity_score`: Pairwise cosine distance across $K=3$ samples from the same face.

---

## 5. Validation Strategy

### Unit Tests
```bash
pytest tests/test_cvae.py -v
```
- `test_reparameterization_gradient`: Assert gradients propagate through sampled latents $\frac{\partial \mathcal{L}}{\partial \mu} \ne 0$ and $\frac{\partial \mathcal{L}}{\partial \sigma} \ne 0$.
- `test_kl_free_bits_behavior`: Assert that when KL per dimension is below $\lambda_{\text{fb}}$, the computed loss equals $\lambda_{\text{fb}} \times 192$ and its backward gradient is strictly zero.
- `test_zero_beta_warmup`: Assert that at step 0, $\beta = 0$, and the optimization updates weights purely via reconstruction loss.

### Smoke Test
- Run 5 epochs on a 5,000-sample partition of VoxCeleb2 using `scripts/run_training.py --smoke-test`.
- Verify loss strictly decreases, execution completes in $<3$ minutes, and GPU VRAM remains stable.

---

## 6. Acceptance Criteria

1. **Reconstruction Convergence:** Mean validation cosine similarity $\cos(\hat{e}_s, e_s)$ on held-out validation pairs reaches $\ge 0.72$ (cosine loss $\le 0.28$).
2. **Posterior Collapse Prevention:** Active latent units ($\text{KL}_d > 0.1$ nats) $\ge 32$ out of 192 dimensions after full training.
3. **Calibrated Diversity Score:** Pairwise cosine distance across $K=3$ sampled voices for the same face falls within $[0.15, 0.40]$ (non-zero variance, but constrained within plausible human identity bounds).
4. **Deterministic Baseline Superiority:** Validation cosine similarity exceeds that of a simple mean-speaker predictor by at least $15\%$.
5. **Memory & Speed Performance:** Full training epoch (100,000 pairs, batch size 64) finishes in $\le 8$ minutes on an RTX 4050 Laptop GPU with peak VRAM $<2.2\text{ GB}$.

---

## 7. Risks & Trade-offs

| Risk | Impact | Mitigation Strategy |
|---|---|---|
| Posterior collapse despite free bits | Degenerates to deterministic model | Reduce $\beta_{\text{max}}$ from $0.5$ to $0.1$; increase $\lambda_{\text{fb}}$ to $1.0$ nats |
| Over-regularization suppresses diversity | Samples have zero diversity ($DS < 0.05$) | Adjust annealing schedule to cyclical annealing (Fujimoto et al.) |
| Outlier face vectors cause unbounded gradients | Training NaN instability | Enforce LayerNorm on face conditioning and clamp log-variance to $[-10, 2]$ |

---

## 8. Deliverables

- `voix/models/cvae.py`: CVAE model classes (`CVAE`, `CVAEEncoder`, `CVAEPrior`, `CVAEDecoder`).
- `voix/training/losses.py`: `CVAELoss` with cosine reconstruction and free-bits KL.
- `voix/training/trainer_cvae.py`: Complete training manager with W&B logging.
- `configs/cvae_training.yaml`: Canonical hyperparameters.
- `scripts/run_training.py`: CLI training script.
- `checkpoints/cvae_best.pt`: Saved model checkpoint satisfying all acceptance criteria.

---

## 9. Documentation Updates

- Update `project-context/architecture.md` Section 2 with empirical loss convergence values.
- Document best hyperparameter configuration in `project-context/mvp.md`.

---

## 10. Dependencies

- **Predecessor:** `PHASE-2` (Feature Extraction & Data Pipelines).
- **Successor:** `PHASE-4` (Acoustic Synthesis & Style Adapter Integration).
