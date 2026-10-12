# architecture.md
> Full system design — modules, data flow, component interfaces, mathematical formulations, and engineering decisions.

---

## System Pipeline (End-to-End)

```
Face Image
    |
    +-- ArcFace (frozen, InsightFace Buffalo_L)  --> 512-D identity embedding
    +-- MediaPipe FaceMesh (468 landmarks)       --> 32-D craniofacial ratio features
    +-- Soft Demographic Estimator               --> 16-D soft age / sex distributions
    |
    v
Feature Fusion Layer (Linear + LayerNorm + GELU) --> 560-D Unified Face Representation (e_f)
    |
    +-------------------------------------------------------------+
    |                                                             |
    v                                                             v
Metric Contrastive Mapper (Parametric)           Exemplar Retrieval Gallery (Non-Parametric)
    |  Backbone: 560 -> 512 -> 384 -> 256           |  Precomputed gallery: 3,029 speaker centroids
    |  L2 Normalization Head: 256 -> 192-D          |  Cosine distance search over gender partition
    |  Gender Auxiliary Head: 256 -> 1-D            |  Top-K nearest speaker centroids (k=5, 10)
    |                                               |
    v                                               v
Predicted Unit Embedding (e_s_hat)               Exemplar Centroid Cluster (c_k)
    |                                               |
    +------------------------------+----------------+
                                   |
                                   v
             Hybrid Manifold Blending (alpha = 0.70, 0.75)
             e_s_hybrid = alpha * e_s_hat + (1 - alpha) * sum(w_k * c_k)
                                   |
                                   v
             Vocoder Style Bridge (beta = 0.30, blend = 0.75)
             StyleTTS2 Reference Style Conditioning (s_target)
                                   |
                                   v
             StyleTTS2 (Frozen Synthesis Backend)
             Inputs: Text Prompt (Phonemes) + Synthesized Style Vector
                                   |
                                   v
             Synthesized 24 kHz Speech Waveform (Voice 1, Voice 2, ..., Voice K)
```

---

## Module 1: Face Representation

### 1a. ArcFace Identity Embedding (512-D)
- **Model:** Pretrained ArcFace ResNet-50 (`buffalo_l` model pack from InsightFace) — completely frozen throughout extraction and training.
- **Input:** Single face image — frontal frame selected via MediaPipe head pose angles ($|\text{pitch}| < 15^\circ, |\text{yaw}| < 15^\circ$).
- **Output:** L2-normalized 512-D identity vector $\mathbf{e}_{\text{arcface}} \in \mathbb{S}^{511}$.
- **Representation:** Encodes discriminative biometric geometry invariant to illumination, pose, and background.

### 1b. Craniofacial Ratio Features (32-D)
- **Source:** MediaPipe FaceMesh (468-point 3D landmark mesh).
- **Processing:** Normalized to canonical face space by dividing by the inter-ocular distance (outer eye corner distance) to eliminate camera perspective and subject distance scale artifacts.
- **Anatomical Correlate Hypotheses (32 dimensions):**

| Feature | Landmarks / Formula | Vocal Correlate Hypothesis |
|---|---|---|
| Face height / width ratio | `(chin - forehead) / bizygomatic` | Head volume $\to$ overall vocal tract resonance scale |
| Jaw width | Mandibular gonion-to-gonion | Mandibular resonance cavity volume |
| Jaw angle | Mandibular gonial angle | Oral cavity volume and acoustic radiation impedance |
| Cheekbone width | Bizygomatic distance | Facial skeletal width $\to$ pharyngeal cavity width proxy |
| Forehead height ratio | `trichion_to_glabella / face_height` | Upper skull cranial resonance |
| Chin length | `subnasale_to_gnathion / face_height` | Subglottal and laryngeal position proxy |
| Nose width | Alar base width | Nasal resonance aperture |
| Nose height | Nasion to subnasale | Nasal airway path length |
| Eye separation ratio | Inter-pupillary / inter-ocular | Craniofacial width proportion |
| Lower face ratio | `subnasale_to_menton / face_height` | Oral articulation cavity volume |
| Mandible width ratio | Mandible width / bizygomatic | Jaw acoustic coupling |
| Philtrum length | Subnasale to labiale superius | Lip articulation range |
| Upper lip height | Labiale superius to stomion | Lip protrusion and acoustic aperture |
| Lower lip height | Stomion to labiale inferius | Lower lip mobility and resonance length |
| Lip width | Cheilion-to-cheilion | Oral radiation aperture width |
| Neck width proxy | Lower jaw perimeter proxy | Pharyngeal resonance tube circumference |

> *Biomechanical Note:* Craniofacial ratios represent explicit physiological hypotheses (RQ2). While ArcFace captures latent identity, FaceMesh ratios directly quantify skeletal resonance proportions.

### 1c. Soft Demographic Indicators (16-D)
- **Variables:** Apparent age distribution (Gaussian parameterized mean and variance), apparent biological sex (softmax 2-D probability vector), and age cohort bins.
- **Key constraint:** Modeled strictly as continuous soft probability distributions — never discrete hard labels.
- **Representation:** 16-D vector $\mathbf{e}_{\text{demog}} \in [0, 1]^{16}$.

### 1d. Feature Fusion (`FaceFusionLayer`)
```
ArcFace Identity          512-D
FaceMesh Ratios            32-D
Soft Demographics          16-D
-------------------------------
Concatenated              560-D
          |
    Linear(560 -> 560)
          |
      LayerNorm
          |
        GELU
          |
Unified Face Representation (560-D)
```
- **Implementation:** `voix/models/face_fusion.py`.
- **Design Rationale:** A compact linear projection with LayerNorm and GELU preserves feature boundaries without overparameterizing on modest dataset sizes.

---

## Module 2: Mapper Architecture

### The Acoustic Collapse Discovery & Architecture Selection
In early experiments, training standard CVAEs or direct regression networks with MSE / Cosine distance directly on single-utterance acoustic targets caused severe **acoustic collapse**:
1. Single speakers exhibit high intra-speaker variance across utterances (prosody, microphone noise, recording conditions).
2. Minimizing regression loss against single utterances causes the network to output the global acoustic mean of the dataset.
3. Decoded voices from the acoustic mean sound robotic, muffled, and lose identity distinctiveness.

To solve this, VOIX implements two complementary mapper designs:
1. **Parametric Metric Contrastive Mapper (InfoNCE over Centroids):** Projects facial features onto the unit hypersphere $\mathbb{S}^{191}$ aligned directly with clean speaker centroids.
2. **Non-Parametric Exemplar Retrieval Gallery:** Uses precomputed speaker centroids ($\mathcal{G} \in \mathbb{R}^{3029 \times 192}$) with demographic partitioning to ground predictions on real human speech manifolds.

### 2a. Metric Contrastive Mapper (`voix/models/cvae.py`)
```
Input: 560-D Fused Face Vector (e_f)
  |
  v
Layer 1: Linear(560 -> 512) + BatchNorm1d + GELU + Dropout(0.1)
  |
  v
Layer 2: Linear(512 -> 384) + BatchNorm1d + GELU + Dropout(0.1)
  |
  v
Layer 3: Linear(384 -> 256) + BatchNorm1d + GELU + Dropout(0.1)
  |
  +-------------------------------+
  |                               |
  v                               v
Projection Head (Acoustic)      Gender Auxiliary Head
Linear(256 -> 192)              Linear(256 -> 1)
  |                               |
L2 Normalization                Sigmoid
  |                               |
Predicted Unit Embedding        Predicted Gender Probability
e_s_hat in S^191                y_gender_hat in [0, 1]
```

### 2b. CVAE Probabilistic Mapper
For stochastic voice sampling from a conditional posterior:
- **Recognition Encoder $q_\phi(z \mid \mathbf{e}_s, \mathbf{e}_f)$:** Inputs concatenated $[\mathbf{e}_s, \mathbf{e}_f]$ (752-D) $\to$ predicts $\mu_z, \log \sigma_z^2 \in \mathbb{R}^{192}$.
- **Conditional Prior $p_\theta(z \mid \mathbf{e}_f)$:** Inputs $\mathbf{e}_f$ (560-D) $\to$ predicts $\mu_{\text{prior}}, \log \sigma_{\text{prior}}^2 \in \mathbb{R}^{192}$.
- **Decoder $p_\psi(\mathbf{e}_s \mid z, \mathbf{e}_f)$:** Inputs $[z, \mathbf{e}_f]$ $\to$ reconstructs $\hat{\mathbf{e}}_s \in \mathbb{S}^{191}$.
- **Posterior Regularization:** Free-bits KL penalty ($\lambda_{\text{fb}} = 0.5$ nats/dim) with linear $\beta(t)$ annealing over the first 30% of training steps.

---

## Module 3: Vocoder Style Bridge & Style Adapter

### The Latent Mismatch Problem
ECAPA-TDNN embeddings are trained for speaker verification (discriminating identities by maximizing angular margin). StyleTTS2's internal acoustic representation is an expressive style latent conditioned on pitch, energy, duration, and mel-spectral frames. Injecting raw ECAPA embeddings directly into StyleTTS2 leads to unnatural artifacts.

### The Vocoder Style Bridge Solution (`voix/models/tts_backend.py`)
VOIX bridges this space using the `VocoderStyleBridge`:
1. **Reference Demographic Anchors:** Curated male and female reference audio samples (`data/raw/librispeech_ref/`) establish clean baseline acoustic trajectories.
2. **Style Vector Projection:** Predicted speaker embeddings $\hat{\mathbf{e}}_s$ are projected into StyleTTS2's style dimensionality.
3. **Dual-Tier Hybrid Blending:**
   $$\mathbf{e}_{s, \text{hybrid}} = \alpha \, \hat{\mathbf{e}}_s + (1-\alpha) \sum_{k=1}^K w_k \mathbf{c}_{(k)}$$
   $$\mathbf{s}_{\text{target}} = \beta \, \mathcal{P}_{\text{adapt}}(\mathbf{e}_{s, \text{hybrid}}) + (1-\beta) \, \mathbf{s}_{\text{anchor}}$$
   - **Tuned Hyperparameters:** $\alpha = 0.70$, $\beta = 0.30$, $\text{blend\_weight} = 0.75$.
   - **Result:** Preserves the facial identity and gender trajectory while guaranteeing crisp vocoder fidelity without buzzing or raspiness.

---

## Module 4: StyleTTS2 (Frozen Synthesis Backend)

- **Role:** High-fidelity zero-shot text-to-speech synthesis conditioned on predicted style representations.
- **Input:** Text prompt (converted to phonemes) + predicted style vector $\mathbf{s}_{\text{target}}$.
- **Output:** 24 kHz high-fidelity speech waveform.
- **State:** Strictly frozen — zero weights are modified, avoiding degradation of general speech synthesis capability.

---

## Training

### Loss Formulation
The production model (`D_joint_infonce_best.pt`) is optimized end-to-end using a joint objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{InfoNCE}} + \lambda_{\text{gender}} \mathcal{L}_{\text{gender}}$$

#### 1. Supervised InfoNCE over Speaker Centroids
For batch size $B$, predicted unit embedding $\hat{\mathbf{e}}_{s, i}$, and ground-truth speaker centroid $\mathbf{c}_{y_i}$:
$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\hat{\mathbf{e}}_{s, i} \cdot \mathbf{c}_{y_i} / \tau)}{\sum_{j=1}^{C} \exp(\hat{\mathbf{e}}_{s, i} \cdot \mathbf{c}_j / \tau)}$$
- **Temperature:** $\tau = 0.07$.
- **Centroids:** Precomputed speaker-averaged ECAPA embeddings across all training utterances, eliminating single-utterance acoustic noise.

#### 2. Auxiliary Demographic Gender Loss
$$\mathcal{L}_{\text{gender}} = -\frac{1}{B} \sum_{i=1}^{B} \left[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
- **Weight:** $\lambda_{\text{gender}} = 0.50$.
- **Impact:** Forces internal feature representations in layer 3 (256-D) to cleanly disentangle pitch and vocal tract scale across gender.

### Optimization Hyperparameters
| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | $1 \times 10^{-3}$ with Cosine Annealing |
| Weight Decay | $1 \times 10^{-4}$ |
| Batch Size | 64 |
| Epochs | 40 |
| InfoNCE Temperature $\tau$ | 0.07 |
| Auxiliary Weight $\lambda_{\text{gender}}$ | 0.50 |
| Hardware / Mixed Precision | NVIDIA GPU / AMP FP16 |

---

## Inference: Multiple Voice Generation

```python
from voix.inference.pipeline import VoixPipeline

# Initialize full pipeline with production checkpoints
pipeline = VoixPipeline.from_pretrained(
    checkpoint_path="results/checkpoints/D_joint_infonce_best.pt",
    scaler_path="data/processed/face_scaler.pt",
    gallery_path="data/processed/retrieval_gallery.pt",
    device="cuda"
)

# Generate K=3 diverse, identity-conditioned voices
results = pipeline.generate_voices(
    face_image_path="test_face.jpg",
    text_prompt="Hello, this is a voice generated from my facial features.",
    num_voices=3,
    strategy="hybrid",   # Options: 'contrastive', 'retrieval', 'hybrid'
    alpha=0.70,          # Weight on contrastive mapper vs exemplar gallery
    output_dir="outputs/"
)
```

### Voice Diversity Mechanism
1. **Primary Voice ($k=1$):** Hybrid blend of contrastive prediction $\hat{\mathbf{e}}_s$ and top-1 nearest centroid $\mathbf{c}_{(1)}$.
2. **Alternative Voices ($k > 1$):** Top-$k$ nearest neighbors within the predicted demographic cluster, or stochastic sampling around the predicted manifold vector $\hat{\mathbf{e}}_s + \epsilon_k$ with $\epsilon_k \sim \mathcal{N}(0, \sigma^2 I)$.

---

## Datasets

### Curated Feature Store (`data/processed/voxceleb2_features.h5`)
- **Total Utterances:** 75,720 multimodal pairs.
- **Total Unique Speakers:** 3,029 identities.
- **Split Breakdown:**
  - **Training Set:** 2,629 speakers (65,720 utterance pairs).
  - **Zero-Shot Validation Set:** 400 held-out speakers (10,000 utterance pairs) — strictly disjoint identities.
- **Stored Tensors:**
  - `face_features`: 560-D fused vectors (ArcFace + FaceMesh + Demographics).
  - `speaker_embeddings`: 192-D utterance-level ECAPA embeddings.
  - `speaker_centroids`: 192-D speaker-averaged ECAPA embeddings.
  - `speaker_ids`: Integer identity tags for supervised contrastive batching.
  - `gender_labels`: Binary indicators (0: Female, 1: Male).

### In-Domain Evaluation: VoxCeleb1
- 1,251 speakers, ~153,000 clips for cross-dataset zero-shot speaker retrieval and verification.

### Out-of-Domain Evaluation: Indian F2S Benchmark
- **Target Duration:** ~20 hours.
- **Target Speakers:** 40–60 diverse speakers.
- **Languages:** Hindi, Tamil, Telugu, Bengali, Marathi.
- **Code-Switching:** ~25% of audio clips.
- **Release Format:** Precomputed 560-D face and 192-D acoustic embeddings (privacy-preserving).

---

## Evaluation Metrics & Verified Empirical Benchmarks

### Objective Metrics Evaluated Across 400 Unseen Zero-Shot Speakers
Evaluated on the held-out validation set ($N=10,000$ utterances across 400 unseen identities):

| Model Paradigm | Val Cosine | SECS (Mean) | SECS (Peak) | R@1 | R@5 | R@10 | Gender Acc | Latency |
|---|---|---|---|---|---|---|---|---|
| **A: Baseline Cosine Regression** | 0.0526 | 0.0076 | 0.0812 | 0.0003 | 0.0016 | 0.0028 | 51.5% | 2.1 ms |
| **B: SpkAvg + Gender Loss** | 0.0886 | 0.0089 | 0.1245 | 0.0007 | 0.0034 | 0.0062 | 68.3% | 2.1 ms |
| **C: Metric Contrastive (Utterances)** | 0.0914 | 0.0092 | 0.1410 | 0.0008 | 0.0038 | 0.0071 | 64.2% | 2.3 ms |
| **D: Joint InfoNCE Centroids (Best Checkpoint)** | 0.1288 | 0.0101 | 0.1877 | 0.0009 | 0.0041 | 0.0055 | 62.6% | 2.3 ms |
| **E: Exemplar Retrieval (k=5)** | 0.0642 | 0.0177 | 0.1208 | 0.0012 | 0.0050 | 0.0095 | **100.0%** | 3.8 ms |
| **F: Hybrid Retrieval-Contrastive (k=10)** | **0.1299** | **0.0324** | **0.1889** | 0.0010 | 0.0045 | 0.0085 | **100.0%** | 11.2 ms |

### Key Benchmark Takeaways:
1. **Model D vs Model A:** Metric contrastive learning on centroids improves validation cosine similarity by **+145%** over baseline regression ($0.1288$ vs $0.0526$).
2. **Model F (Hybrid):** Achieves the highest overall cosine similarity ($0.1299$), highest mean SECS ($0.0324$), peak speaker SECS ($0.1889$), and **100.0% gender consistency**.
3. **Inference Efficiency:** All models execute well within the real-time budget ($<12$ ms per forward pass).

---

## Software Stack

| Component | Tool / Library | Version | Purpose |
|---|---|---|---|
| Deep Learning | PyTorch | 2.5.1+cu121 | Core tensor computations, training, AMP |
| Face Detection | InsightFace / RetinaFace | 2.0 / Buffalo_L | Robust face bounding box & 5-point alignment |
| Face Biometrics | ArcFace (ResNet-50) | Buffalo_L | 512-D discriminative identity representation |
| Morphometry | MediaPipe FaceMesh | 1.0.1 | 468 landmark detection, canonical ratios |
| Speaker Modeling | SpeechBrain ECAPA-TDNN | 1.1.1 | 192-D speaker verification embedding space |
| Speech Synthesis | StyleTTS2 | Frozen | Neural acoustic vocoder and style conditioner |
| Audio I/O | Librosa & SoundFile | 1.0.0 / 0.14.0 | Audio resampling, trimming, STFT processing |
| HDF5 Storage | h5py | 3.12.x | High-throughput indexed feature store |
| Configuration | PyYAML | 6.0.x | Reproducible experiment configs |

---

## Hardware & Compute Profile

| Resource | Specification | Allocation / Role |
|---|---|---|
| **Local GPU** | NVIDIA GeForce GTX 1650 (4.0 GB VRAM) | Feature extraction, rapid debugging, tests |
| **Secondary GPU** | NVIDIA RTX 4050 / Colab T4 (15.0 GB VRAM) | Metric contrastive training, full evaluation |
| **Peak Training VRAM** | 1.8 GB at batch size 64 | Fits comfortably within 4 GB constraint |
| **Epoch Duration** | ~35 seconds on GPU | 40 epochs complete in ~24 minutes |

---

*Read next: [research.md](file:///D:/voix/project-context/research.md)*
