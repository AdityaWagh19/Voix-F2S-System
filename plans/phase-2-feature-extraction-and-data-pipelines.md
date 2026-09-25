# Phase 2: Feature Extraction & Data Pipelines

> **Phase Identifier:** `PHASE-2`  
> **Target Duration:** Weeks 2–3  
> **Status:** Pending Execution  
> **Prerequisites:** `PHASE-1` Complete

---

## 1. Objective

Design, implement, and validate the offline multimodal feature extraction pipeline. Construct a paired dataset of 100,000 samples from VoxCeleb2 (2,000 speakers $\times$ 50 utterances) consisting of a 560-D unified facial conditioning vector $e_f$ and a 192-D utterance-level ECAPA-TDNN speaker embedding $e_s$. Serialize the extracted representations into high-throughput HDF5/Safetensors archives with metadata validation.

---

## 2. Scope

### In Scope
- Frontal frame selection module: pitch, yaw, and roll estimation via MediaPipe FaceMesh to pick the most neutral, frontal face frame per video clip.
- 32-D craniofacial geometric ratio extraction: computing 15 primary anatomical distance ratios and angles normalized by inter-ocular distance.
- 16-D soft demographic prior module: probabilistic age (mean + variance) and sex distributions.
- 560-D Linear Feature Fusion layer with LayerNorm and GELU.
- Utterance-level 192-D ECAPA-TDNN extraction across 100,000 VoxCeleb2 audio clips.
- PyTorch `Dataset` and `DataLoader` implementations supporting high-speed memory-mapped tensor access.
- Data validation suite: automated NaN/Inf checks, distribution normality checks, and PCA/UMAP cluster verification.

### Out of Scope
- Training the CVAE generative model (deferred to Phase 3).
- Fine-tuning the feature extractors (all extractors remain strictly frozen).
- Ingesting Indian streaming video data (deferred to Phase 5).

---

## 3. Design Decisions & Rationale

### 3.1 Frontal Frame Selection vs. Frame Averaging
- **Decision:** Select a single optimal frontal, neutral frame per utterance rather than temporal feature averaging across video frames.
- **Rationale:** Averaging face embeddings across speech introduces articulatory motion artifacts (open mouths, squinting, jaw shifts during vowels) which degrade static craniofacial ratio measurements. Selecting the frame with minimum absolute yaw and pitch ($|\text{yaw}| < 10^\circ, |\text{pitch}| < 10^\circ$) yields clean, invariant skeletal geometry.

### 3.2 Inter-Ocular Normalization for Craniofacial Ratios
- **Decision:** All 3D landmark coordinates from MediaPipe are scaled by the inter-ocular distance (Euclidean distance between left eye outer canthus #33 and right eye outer canthus #263) prior to computing distance ratios.
- **Rationale:** Inter-ocular normalization renders distance ratios invariant to camera focal length, face-to-camera distance, and digital zoom.

### 3.3 Utterance-Level vs. Speaker-Averaged Acoustic Embeddings
- **Decision:** Extract ECAPA-TDNN embeddings per individual 3–10s utterance rather than averaging embeddings across all utterances of a speaker.
- **Rationale:** If the target $e_s$ is speaker-averaged, within-speaker variance is zero, causing the generative CVAE to collapse its predictive variance $\sigma_\theta^2 \to 0$ (degenerating into a deterministic model). Utterance-level targets preserve intra-speaker acoustic variation (prosodic state, emotional pitch, room acoustics), allowing the CVAE to learn a calibrated variance distribution.

### 3.4 Data Format: HDF5 / Safetensors
- **Decision:** Store extracted embedding arrays in `.h5` or `.safetensors` files indexed by `(speaker_id, utterance_id)` alongside a unified `metadata.parquet`.
- **Rationale:** Parsing 100,000 small files on disk causes catastrophic I/O bottlenecks and Windows NTFS slowdowns. Memory-mapped HDF5 provides $>1,500$ samples/sec throughput during training.

### 3.5 Dataset Sourcing & Cloud Extraction Architecture
- **Context:** Official direct download links on Oxford VGG's website are deprecated due to privacy/GDPR compliance.
- **Sourcing Strategy:**
  1. *Academic Torrents / Kaggle / OpenSLR:* Active mirrors host VoxCeleb1 and VoxCeleb2 data/metadata.
  2. *Cloud-Native Batch Extraction:* Rather than downloading 300+ GB of raw videos locally to an entry-level GPU (GTX 1650), batch extraction is packaged as a Google Colab / Kaggle script that ingests the raw videos in cloud ephemeral storage and saves the final compact `~280 MB` HDF5 file directly to Google Drive.
  3. *Local Prototyping:* A compact 50–100 speaker slice (or VoxCeleb1 subset) is used for rapid local verification.

---

## 4. Sequential Implementation Tasks

```
[T2.1] Frontal Frame Selection & Head-Pose Filtering
       ├── MediaPipe pose angle calculator (pitch, yaw, roll)
       └── Optimal frame index selector for video clips

[T2.2] Craniofacial Morphological Ratio Calculator (32-D)
       ├── Inter-ocular canonical normalizer
       ├── Implementation of 32 anatomical ratios
       └── Unit test against synthetic landmark grids

[T2.3] Soft Demographic Estimator (16-D)
       ├── Probabilistic age and sex head implementations
       └── Gradient isolation assertion

[T2.4] Unified Multimodal Feature Fusion Layer (560-D)
       ├── Concatenation: 512-D + 32-D + 16-D
       ├── Linear(560 -> 560) + LayerNorm + GELU
       └── Shape & gradient assertion test

[T2.5] Utterance-Level ECAPA-TDNN Audio Pipeline
       ├── Audio resampler (16 kHz mono WAV) & silence trimmer
       └── SpeechBrain batch feature extractor

[T2.6] Batch Extraction Pipeline & Storage Engine
       ├── Multi-threaded extractor runner for VoxCeleb2
       ├── HDF5 serialization with metadata.parquet
       └── Google Drive synchronization script

[T2.7] Dataset Class & Statistical Verification
       ├── PyTorch VoxCelebPairedDataset implementation
       ├── Automated NaN / Inf / boundary sanity audit
       └── PCA / UMAP visualization script for speaker clustering
```

### Detailed Task Specifications

#### Task 2.1: Frontal Frame Selection
- File: `voix/data/face_extractor.py`
- Extract 3D head pose from MediaPipe 468 landmarks:
  - Nose tip (#1), Chin (#199), Left eye outer (#33), Right eye outer (#263), Left mouth corner (#61), Right mouth corner (#291).
  - Solve PnP (Perspective-n-Point) using standard 3D facial model to extract rotation vector $\to$ Euler angles (pitch, yaw, roll).
- Define selection metric: $S_t = |\text{yaw}_t| + 1.5|\text{pitch}_t| + 0.5|\text{roll}_t| + \text{mouth\_openness}_t$.
- Return frame $t^* = \arg\min_t S_t$.

#### Task 2.2: Craniofacial Ratio Calculator (32-D)
- File: `voix/data/morphology.py`
- Landmarks mapped to anatomical definitions:
  - Inter-ocular distance: $d_{\text{eye}} = \|\mathbf{p}_{33} - \mathbf{p}_{263}\|_2$.
  - Face height: $\|\mathbf{p}_{10} - \mathbf{p}_{152}\|_2 / d_{\text{eye}}$.
  - Jaw width: $\|\mathbf{p}_{234} - \mathbf{p}_{454}\|_2 / d_{\text{eye}}$.
  - Nose width: $\|\mathbf{p}_{102} - \mathbf{p}_{331}\|_2 / d_{\text{eye}}$.
  - Mandible angle: Angle between vector $(\mathbf{p}_{152} - \mathbf{p}_{234})$ and $(\mathbf{p}_{152} - \mathbf{p}_{454})$.
  - Additional 27 distance and angular ratios specified in `architecture.md`.
- Output: Float32 vector of length exactly 32.

#### Task 2.3: Soft Demographic Estimator (16-D)
- File: `voix/data/demographics.py`
- Predict soft probability distribution from ArcFace feature:
  - Apparent age: Gaussian mean $\mu_{\text{age}} \in [0, 1]$ and log-variance $\sigma_{\text{age}}^2$ (normalized by 100 years).
  - Apparent biological sex: Softmax 2-D vector $[p_{\text{male}}, p_{\text{female}}]$.
  - Vocal tract length proxy: 4-D soft distribution.
  - Soft categorical age bins: 8-D distribution (child, teen, young_adult, adult, middle_aged, senior, etc.).
- Output: 16-D float32 tensor summing to calibrated probability mass.

#### Task 2.4: Unified Feature Fusion Layer
- File: `voix/models/fusion.py`
- Class `FaceFusionLayer(nn.Module)`:
  ```python
  class FaceFusionLayer(nn.Module):
      def __init__(self, in_features=560, out_features=560):
          super().__init__()
          self.proj = nn.Linear(in_features, out_features)
          self.norm = nn.LayerNorm(out_features)
          self.act = nn.GELU()
      def forward(self, e_id, e_geo, e_demo):
          x = torch.cat([e_id, e_geo, e_demo], dim=-1)
          return self.act(self.norm(self.proj(x)))
  ```

#### Task 2.5 & 2.6: Batch Extraction & HDF5 Serialization
- File: `scripts/extract_voxceleb.py`
- Input: Directory of VoxCeleb2 `.mp4` video files.
- Process:
  1. Read video using OpenCV / PyAV, apply Task 2.1 to select $t^*$.
  2. Crop and align face to $112 \times 112$, pass to ArcFace $\to e_{\text{id}} \in \mathbb{R}^{512}$.
  3. Extract landmarks from $t^*$, pass to Task 2.2 $\to e_{\text{geo}} \in \mathbb{R}^{32}$.
  4. Estimate demographics via Task 2.3 $\to e_{\text{demo}} \in \mathbb{R}^{16}$.
  5. Extract 16 kHz audio via `ffmpeg`, compute ECAPA embedding $\to e_s \in \mathbb{R}^{192}$.
  6. Write tensors into `data/processed/voxceleb2_train.h5` under datasets:
     - `/face_features`: float32 array $(100000, 560)$
     - `/speaker_features`: float32 array $(100000, 192)$
     - `/speaker_ids`: int32 array $(100000,)$

#### Task 2.7: PyTorch Dataset & Integrity Audit
- File: `voix/data/dataset.py`
- Class `VoxCelebPairedDataset(torch.utils.data.Dataset)`:
  - Memory-maps `data/processed/voxceleb2_train.h5`.
  - Supports `__getitem__(idx)` returning `(e_f, e_s, speaker_id)`.
- File: `scripts/verify_dataset.py`:
  - Assert zero NaN, Inf, or constant-zero vectors across 100K rows.
  - Compute silhouette scores on $e_{\text{id}}$ and $e_s$ grouped by `speaker_id`.
  - Export UMAP 2D projections to `artifacts/voxceleb_clusters.png`.

---

## 5. Validation Strategy

### Unit Tests
```bash
pytest tests/test_features.py -v
```
- `test_frontal_frame_selection`: Assert selector chooses frame with minimal yaw/pitch from a 30-frame synthetic rotating sequence.
- `test_craniofacial_ratios`: Assert 32-D output vector is scale-invariant (scaling input image by $2\times$ results in $L_\infty < 10^{-4}$ change in ratios).
- `test_fusion_layer`: Assert output shape is `(B, 560)` and gradients backpropagate cleanly to projection weights.

### Dataset Integrity Audit
```bash
python scripts/verify_dataset.py --h5-path data/processed/voxceleb2_train.h5
```

---

## 6. Acceptance Criteria

1. **Extraction Scale:** Exactly 100,000 paired samples extracted from VoxCeleb2 (2,000 unique identities, 50 utterances each).
2. **Dimension Integrity:**
   - Facial feature tensor shape strictly $(100000, 560)$, float32.
   - Speaker feature tensor shape strictly $(100000, 192)$, float32.
3. **Data Purity:** Zero NaN, zero Inf, and zero null values across all extracted arrays.
4. **Scale Invariance:** Craniofacial ratios demonstrate $<0.01\%$ variation under synthetic affine camera zoom transformations.
5. **Acoustic Speaker Clustering:** UMAP/Silhouette score on 192-D ECAPA embeddings yields silhouette score $>0.35$ with respect to true speaker identities.
6. **I/O Throughput:** `DataLoader` with `batch_size=128` achieves $>2,000$ paired samples per second on local SSD.

---

## 7. Risks & Trade-offs

| Risk | Impact | Mitigation Strategy |
|---|---|---|
| Videos with extreme head occlusion or missing face | Missing frames / crash | Fallback to center frame crop; if no face detected by RetinaFace, drop utterance and replace with next valid clip from same speaker |
| High disk usage from raw video storage | Storage saturation | Use streaming extraction and delete temporary WAV/frame buffers immediately after embedding extraction |
| Demographic estimator bias | Model distortion | Soft probability distributions with entropy regularization; gradient down-weighting in CVAE |

---

## 8. Deliverables

- `voix/data/face_extractor.py`: Frontal frame selector and ArcFace wrapper.
- `voix/data/morphology.py`: 32-D craniofacial ratio extractor.
- `voix/data/demographics.py`: 16-D soft demographic prior estimator.
- `voix/models/fusion.py`: 560-D linear fusion network.
- `voix/data/dataset.py`: PyTorch `VoxCelebPairedDataset` loader.
- `scripts/extract_voxceleb.py`: Batch extraction CLI.
- `scripts/verify_dataset.py`: Statistical validation and UMAP cluster generator.
- `data/processed/voxceleb2_train.h5`: 100K paired tensor archive.

---

## 9. Documentation Updates

- Document dataset schema in `project-context/architecture.md` Section 8.
- Record dataset extraction throughput and silhouette scores in `project-context/tasks.md`.

---

## 10. Dependencies

- **Predecessor:** `PHASE-1` (Environment & Pretrained Backbones).
- **Successor:** `PHASE-3` (Probabilistic CVAE Mapper Development).
