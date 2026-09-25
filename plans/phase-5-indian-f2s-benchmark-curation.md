# Phase 5: Indian F2S Benchmark Curation

> **Phase Identifier:** `PHASE-5`  
> **Target Duration:** Weeks 9–11  
> **Status:** Pending Execution  
> **Prerequisites:** `PHASE-1` Complete (can execute in parallel with Phase 3/4)

---

## 1. Objective

Curate, filter, transcribe, and standardize the India F2S Benchmark dataset — a 20-hour multimodal corpus spanning 40–60 verified speakers across five Indian languages (Hindi, Tamil, Telugu, Bengali, Marathi) with 25% verified code-switching between regional languages and English. Implement a streaming, non-persistent ingestion pipeline that demuxes audio-visual data in memory, enforces strict Active Speaker Detection via SyncNet, filters noise via WADA-SNR, verifies cross-frame identity purity with ArcFace, and releases pre-extracted embeddings and metadata manifests complying with HuggingFace Datasets standards.

---

## 2. Scope

### In Scope
- Non-persistent stream processing engine using `yt-dlp` reading educational/CC-BY audiovisual streams (NPTEL, AI4Bharat public domain) directly into a memory buffer without storing raw video files on disk.
- Frame extraction and face cropping via RetinaFace ($224 \times 224$).
- Audio demuxing and resynthesizing to 16 kHz mono PCM WAV via `ffmpeg`.
- Active Speaker Detection (ASD) using SyncNet to eliminate off-screen dubbing and voice-over mismatches (threshold $> 4.5$).
- Identity purity auditing: computing pairwise ArcFace cosine similarity across all candidate face crops within an utterance to discard multi-person conversational scenes.
- Automatic Speech Recognition (ASR) transcription using OpenAI Whisper large-v3.
- Language identification and code-switching quantification using IndicLID (AI4Bharat).
- Signal-to-Noise Ratio (SNR) filtering using WADA-SNR ($> 20\text{ dB}$) and background music rejection.
- Standardized manifest generation: `metadata.csv`, `speakers.json`, and pre-extracted embedding archives (`arcface.safetensors`, `ecapa.safetensors`, `facemesh.safetensors`).

### Out of Scope
- Re-hosting or distributing raw MP4 videos or full-length audio tracks (only embeddings and metadata manifests are published).
- Training the CVAE from scratch exclusively on Indian data (used for zero-shot testing and fine-tuning in Phase 6).

---

## 3. Design Decisions & Rationale

### 3.1 Non-Persistent Memory Ingestion vs. Mass Video Download
- **Decision:** Stream video segments through a volatile in-memory ring buffer, extracting crops, audio, and embeddings immediately, then flushing raw frames from RAM.
- **Rationale:** Storing 20 hours of raw high-definition video requires $>250\text{ GB}$ disk space and raises copyright redistribution issues. In-memory streaming adheres strictly to fair use, eliminates disk saturation on the local workstation, and produces only derived numerical embeddings.

### 3.2 SyncNet Active Speaker Detection Threshold ($> 4.5$)
- **Decision:** Reject any segment where the SyncNet audio-to-video synchronization confidence score falls below $4.5$.
- **Rationale:** Indian academic and lecture videos frequently feature slides or studio cuts where an off-screen narrator speaks while a different presenter's photo or face is displayed. An uncalibrated pipeline would inject false cross-modal pairs. SyncNet guarantees that the lip movements physically match the acoustic energy.

### 3.3 25% Code-Switching Stratification Requirement
- **Decision:** Explicitly mandate that at least 25% of all clips per speaker contain intrasentential code-switching (Indic language mixed with English vocabulary), validated by IndicLID.
- **Rationale:** Code-switching is endemic in Indian linguistic environments. Standard Western speech models fail catastrophically when phonetic systems shift dynamically mid-sentence. Ensuring 25% representation makes the benchmark uniquely valuable for global speech research.

---

## 4. Sequential Implementation Tasks

```
[T5.1] In-Memory Streaming Ingestion Engine
       ├── yt-dlp pipe to OpenCV / ffmpeg subprocess
       └── Volatile ring buffer for 3-10 second video chunks

[T5.2] SyncNet Audio-Visual Synchronizer
       ├── PyTorch SyncNet architecture implementation
       ├── Lip-to-speech cross-correlation scoring function
       └── Calibration audit on sample NPTEL lectures

[T5.3] Cross-Frame Facial Identity Purity Gate
       ├── RetinaFace face detection and 224x224 alignment
       └── Frame-to-frame ArcFace cosine similarity filter (min sim > 0.85)

[T5.4] Phonetic & Linguistic Analysis Engine
       ├── Whisper large-v3 transcription and timestamping
       ├── IndicLID segment language tagging (token-level)
       └── Code-switching index calculator (CS_index >= 0.15)

[T5.5] Acoustic Quality Gating
       ├── WADA-SNR calculator (threshold > 20 dB)
       └── Spectral flatness music rejection filter

[T5.6] Pilot Ingestion & Threshold Calibration
       ├── Process 25 speakers (5 speakers x 5 languages)
       ├── Manual audit of 100 sample clips
       └── Tune SyncNet and SNR thresholds

[T5.7] Full Corpus Extraction Pipeline
       ├── Batch execution across 40-60 target identities
       └── Extract ArcFace (512-D), ECAPA (192-D), and FaceMesh (32-D)

[T5.8] Benchmark Packaging & Schema Serialization
       ├── Generate metadata.csv and speakers.json
       ├── Package embedding arrays into safetensors format
       └── Build HuggingFace Datasets loading script
```

### Detailed Task Specifications

#### Task 5.1: Streaming Ingestion Pipeline
- File: `voix/data/stream_curator.py`
- Stream segments via `yt-dlp`:
  ```python
  import subprocess, io, cv2, numpy as np

  def stream_video_chunk(url: str, start_time: float, duration: float):
      cmd = [
          "ffmpeg", "-ss", str(start_time), "-t", str(duration),
          "-i", url, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"
      ]
      # Stream directly into memory buffer without disk writing
  ```

#### Task 5.2: SyncNet Active Speaker Detection
- File: `voix/data/syncnet.py`
- Model: Pretrained 2D-CNN SyncNet (VGG audio-visual architecture).
- Inputs: 5 consecutive mouth-crop frames ($112 \times 112$) and corresponding 0.2s mel-spectrogram.
- Metric: Confidence $C = \max_d (S_{av}(d)) - \text{mean}(S_{av})$.
- Filter: Pass if $C \ge 4.5$ and offset $|d^*| \le 1$ frame.

#### Task 5.3: Cross-Frame Identity Purity
- File: `voix/data/face_verifier.py`
- Extract ArcFace embedding for face crop in frame 0, frame $N/2$, and frame $N$.
- Compute minimum cosine similarity: $S_{\text{id}} = \min(\cos(e_0, e_{N/2}), \cos(e_{N/2}, e_N))$.
- Discard clip if $S_{\text{id}} < 0.85$ (indicates cut, pan, or speaker transition).

#### Task 5.4: Whisper Transcription & IndicLID Tagging
- Transcribe audio chunk with Whisper large-v3: obtain string $T$ and token timestamps.
- Pass audio through `IndicLID-BERT` per 1-second window.
- Calculate Code-Switching Metric:
  $$\text{CS\_Ratio} = \frac{\text{Tokens}(\text{English})}{\text{Tokens}(\text{Regional}) + \text{Tokens}(\text{English})}$$
- Flag `is_code_switched = True` if $0.15 \le \text{CS\_Ratio} \le 0.85$.

#### Task 5.5: Acoustic Quality Gating
- Calculate WADA-SNR (Waveform Amplitude Distribution Analysis):
  ```python
  def compute_wada_snr(waveform: np.ndarray) -> float:
      # Return SNR in decibels
  ```
- Reject if $\text{SNR} < 20.0\text{ dB}$.
- Reject if spectral centroid variance indicates persistent background music.

#### Task 5.8: HuggingFace Dataset Serialization
- Directory: `data/india_benchmark/`
- Schema for `metadata.csv`:
  ```csv
  clip_id,speaker_id,gender,language,is_code_switched,cs_ratio,snr_db,syncnet_score,duration_sec,text_transcript
  IN_HIN_001_001,IN_HIN_001,female,hindi,True,0.28,24.3,5.82,4.52,"आज का topic neural networks के basic concepts हैं"
  ```
- Schema for `speakers.json`:
  ```json
  {
    "IN_HIN_001": {
      "language": "hindi",
      "gender": "female",
      "source": "NPTEL_DeepLearning",
      "total_duration_sec": 1240.5,
      "clip_count": 250
    }
  }
  ```
- Safetensors archives:
  - `embeddings/arcface_512.safetensors`
  - `embeddings/ecapa_192.safetensors`
  - `embeddings/facemesh_32.safetensors`

---

## 5. Validation Strategy

### Unit Tests
```bash
pytest tests/test_curation.py -v
```
- `test_syncnet_alignment`: Assert SyncNet produces confidence $<2.0$ on artificially misaligned audio/video (+1.0 second desync) and $>5.0$ on aligned inputs.
- `test_indiclid_tagging`: Assert IndicLID correctly flags a synthesized Hindi-English sentence as code-switched.
- `test_wada_snr`: Assert clean speech tests $>25\text{ dB}$ while Gaussian noise-injected speech fails threshold.

### Corpus Verification
```bash
python scripts/audit_indian_benchmark.py --dir data/india_benchmark/
```
- Verify gender parity ($50 \pm 5\%$ female).
- Verify code-switching quota ($\ge 25\%$ code-switched clips).
- Verify language distribution (at least 3.5 hours per language).

---

## 6. Acceptance Criteria

1. **Volume & Identity Scale:** Total accumulated speech duration $\ge 20.0$ hours across $40 \le N \le 60$ distinct speaker identities.
2. **Linguistic Representation:** Five languages represented (Hindi, Tamil, Telugu, Bengali, Marathi), with at least $3.5$ hours of verified speech per language.
3. **Code-Switching Ratio:** Exactly $\ge 25.0\%$ of all curated clips possess confirmed intrasentential code-switching tagged by IndicLID.
4. **Demographic Balance:** Verified biological sex ratio is between $45\%$ and $55\%$ female speakers.
5. **Acoustic & Visual Quality:**
   - 100% of clips pass SyncNet ASD score $>4.5$.
   - 100% of clips exhibit WADA-SNR $>20\text{ dB}$.
   - Cross-frame ArcFace identity purity $>0.85$ on all accepted clips.
6. **Data Governance & Integrity:** Zero raw audiovisual clips stored in distribution bundle; all embeddings exported in `.safetensors` format with complete `metadata.csv` passing Pydantic schema validation.

---

## 7. Risks & Trade-offs

| Risk | Impact | Mitigation Strategy |
|---|---|---|
| Ingesting copyrighted or non-free media | Legal infringement | Restrict URLs strictly to NPTEL (CC-BY) and AI4Bharat approved academic datasets |
| Lip-sync failure on non-frontal lecture poses | High clip rejection rate | Relax SyncNet threshold to $4.0$ exclusively for lecture segments with verified single-speaker audio |
| Low volume in certain regional languages (e.g., Telugu/Marathi) | Unbalanced corpus | Supplement with public domain Parliamentary / All India Radio press briefing recordings |

---

## 8. Deliverables

- `voix/data/stream_curator.py`: In-memory video streaming extractor.
- `voix/data/syncnet.py`: PyTorch SyncNet active speaker detector.
- `scripts/curate_indian_benchmark.py`: End-to-end dataset curation script.
- `scripts/audit_indian_benchmark.py`: Statistical verification script.
- `data/india_benchmark/metadata.csv`: Master metadata manifest.
- `data/india_benchmark/speakers.json`: Speaker demographic summary.
- `data/india_benchmark/embeddings/`: Safetensors embedding archives.

---

## 9. Documentation Updates

- Update `project-context/architecture.md` Section 5 with final dataset statistics.
- Document HuggingFace dataset card in `README.md`.

---

## 10. Dependencies

- **Predecessor:** `PHASE-1` (Environment & Backbones).
- **Successor:** `PHASE-6` (Baselines, Ablations & Empirical Evaluation).
