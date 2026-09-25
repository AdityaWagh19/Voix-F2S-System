# VOIX Comprehensive Literature Survey Framework
**A Modernized Reference Taxonomy for Probabilistic Craniofacial-to-Vocal Synthesis**

* **Project:** VOIX (Probabilistic Craniofacial-to-Vocal Synthesis)
* **Target Corpus:** 25 Papers (Balanced: 40% Modern 2022-2024 SOTA | 30% Standard Tooling 2018-2021 | 30% Seminal Foundations and Biomechanics)
* **Venue Standards:** IEEE (TPAMI, ICASSP, CVPR), ACM (SIGGRAPH, TOG, MM), NeurIPS, ICLR, Interspeech, ECCV, JASA, Journal of Voice
* **Document Version:** 3.0 (Exact 25-Paper Final Confirmed Corpus)

---

## Architectural Alignment and Era Distribution

| Bucket | Domain and Research Focus | VOIX Subsystem Mapped | Papers |
|:---:|---|---|:---:|
| **1** | Cross-Modal Biometrics and Synthesis | Problem Statement and Direct Baselines | 1-4 |
| **2** | Visual Biometrics and 3D Craniofacial Geometry | Module 1: Facial Analysis and FLAME Mesh | 5-8 |
| **3** | Speaker Embeddings, Codecs and Zero-Shot TTS | Module 2 and 4: Acoustic Engine and Vocoding | 9-12 |
| **4** | Probabilistic Generative Modeling | Module 3: Probabilistic Face-to-Voice Adapter | 13-16 |
| **5** | Craniofacial Biomechanics and Vocal Tract Physics | Module 5: Anatomical Priors and Physical Bounds | 17-19 |
| **6** | Audiovisual Corpora, Cross-Modal Identity and Ethics | Module 6, 7 and 8: Data, Benchmarks and Guardrails | 20-25 |

---

## BUCKET 1: Cross-Modal Audio-Visual Association and Synthesis Baselines
> **Research Goal:** Ground the cross-modal problem, establish historical baselines, and survey modern unified multimodal representations and cross-modal diffusion backbones.
> **Mapped Module:** Core Problem Statement and State-of-the-Art Benchmarks

* **Core Research Question:** How have models evolved from deterministic feature regression to multimodal joint embeddings and cross-modal latent diffusion?

### Paper 1: Speech2Face: Learning the Face Behind a Voice
* **Authors:** Tae-Hyun Oh, Tali Dekel, Changil Kim, Inbar Mosseri, William T. Freeman, Michael Rubinstein, Wojciech Matusik
* **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 7531-7540
* **Era / Category:** Seminal Pioneer
* **Scope and Methodology:** Seminal paper in cross-modal voice-to-face reconstruction. Maps pre-trained VGG-Vox audio embeddings to a 4096-d face feature space using L1 regression, reconstructed by a frozen VGG-Face decoder trained on millions of instructional YouTube videos.
* **Flaw Addressed by VOIX:** Operates in reverse (Audio to Face); deterministic regression converges to blurry mean facial templates, ignoring the one-to-many ambiguity inherent in cross-modal biometrics. Produces averaged demographic composites rather than individual-specific faces.

### Paper 2: Seeing Voices and Hearing Faces: Cross-Modal Biometric Matching
* **Authors:** Arsha Nagrani, Samuel Albanie, Andrew Zisserman
* **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 8427-8436
* **Era / Category:** Seminal Pioneer
* **Scope and Methodology:** Establishes the cross-modal biometric matching task: given a still face image and an audio clip, determine if they are from the same person. Introduces 1:2 forced-choice and 1:N retrieval evaluation protocols using joint embedding spaces trained with contrastive and multi-way loss objectives.
* **VOIX Adoption:** Provides VOIX primary cross-modal evaluation protocol to benchmark whether generated voices match the corresponding test face above chance at varying retrieval scales.

### Paper 3: ImageBind: One Embedding Space to Bind Them All
* **Authors:** Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, Ishan Misra
* **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 15180-15190
* **Era / Category:** Modern Multimodal SOTA (2023)
* **Scope and Methodology:** Learns a single joint embedding space across six modalities (image, text, audio, depth, thermal, IMU) using only image-paired data via contrastive learning, without requiring simultaneous all-modality pairings. Emergent cross-modal alignment arises naturally from shared image anchoring.
* **VOIX Takeaway:** Demonstrates that visual and acoustic identity representations can be bound into a shared metric manifold where 3D facial geometry tokens and acoustic speaker embedding tokens align without direct paired supervision.

### Paper 4: Imaginary Voice: Face-Styled Diffusion Model for Text-to-Speech
* **Authors:** Jiyoung Lee, Joon Son Chung, Soo-Whan Chung
* **Venue:** IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023, pp. 1-5
* **Era / Category:** Recent Generative SOTA (2023)
* **Scope and Methodology:** Conditions a latent diffusion model on facial identity embeddings extracted from a still face image to synthesize speech with voice characteristics plausibly matching the speaker appearance. Applies classifier-free guidance to modulate the degree of face-to-voice conditioning strength.
* **VOIX Takeaway:** Confirms that probabilistic generative frameworks outperform direct regression in preserving natural high-frequency acoustic detail and pitch variance. Serves as the closest direct prior work baseline to VOIX, differing in that VOIX uses a CVAE adapter over a 3D biomechanical face prior rather than 2D identity embeddings.

---

## BUCKET 2: Visual Representation: 2D Biometrics and 3D Craniofacial Geometry
> **Research Goal:** Ensure visual feature extraction isolates immutable skeletal bone structure rather than superficial lighting, head pose, or facial expression.
> **Mapped Module:** Module 1: Facial Analysis and Geometric Extraction

* **Core Research Question:** How do we extract metric 3D craniofacial bone morphology from unconstrained monocular in-the-wild video?

### Paper 5: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
* **Authors:** Jiankang Deng, Jia Guo, Niannan Xue, Stefanos Zafeiriou
* **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 4690-4699
* **Era / Category:** Standard Biometric Backbone
* **Scope and Methodology:** Imposes an additive angular margin m on the target logit angle (cos(theta + m)) within a hyperspherical embedding space, maximizing the geodesic separability between identities. Achieves state-of-the-art performance across LFW, CFP-FP, and IJB-C face recognition benchmarks.
* **Role in VOIX:** Primary 2D visual identity extractor (InsightFace/ArcFace 512-d embedding) providing global identity features invariant to illumination, pose, and expression. Its output is concatenated with FLAME shape parameters as the visual conditioning signal to the CVAE adapter.

### Paper 6: Learning a Model of Facial Shape and Expression from 4D Scans (FLAME)
* **Authors:** Tianye Li, Timo Bolkart, Michael J. Black, Hao Li, Javier Romero
* **Venue:** ACM Transactions on Graphics (SIGGRAPH Asia), 36(6), 2017, pp. 194:1-194:17
* **Era / Category:** Seminal 3DMM Pioneer
* **Scope and Methodology:** Disentangled 3D Morphable Model (3DMM) learned from 33,000 scans of 3,800 subjects. Separates head shape (300-d cranial/skeletal bone structure) from facial expressions (100-d) and neck/jaw pose. Linear blend skinning with corrective blendshapes captures realistic deformations.
* **Role in VOIX:** Supplies the core biomechanical prior. VOIX isolates the identity shape parameters to ground acoustic vocal tract estimates in immutable cranial geometry, explicitly discarding transient facial expressions that carry no voice information.

### Paper 7: DECA: Detailed Expression Capture and Animation (Learning an Animatable Detailed 3D Face Model from In-the-Wild Images)
* **Authors:** Yao Feng, Haiwen Feng, Michael J. Black, Timo Bolkart
* **Venue:** ACM Transactions on Graphics (TOG / SIGGRAPH), 40(4), 2021, pp. 1-13
* **Era / Category:** Recent 3D Vision SOTA (2021)
* **Scope and Methodology:** Robust monocular encoder-decoder network regressing FLAME parameters from a single unconstrained 2D image. Achieves per-vertex detail reconstruction using learned UV displacement maps capturing fine-scale facial geometry.
* **Role in VOIX:** Provides the automated 3D facial shape extraction pipeline from still frames of VoxCeleb2 video. VOIX uses the extracted identity shape parameters as the craniofacial morphological prior, discarding expression and pose parameters.

### Paper 8: EMOCA: Emotion Driven Monocular Face Capture and Animation
* **Authors:** Radek Danecek, Michael J. Black, Timo Bolkart
* **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 20311-20322
* **Era / Category:** Recent 3D Vision SOTA (2022)
* **Scope and Methodology:** Extends DECA with an emotion consistency loss that supervises the expression code against perceptual emotion classifiers, improving reconstruction fidelity on in-the-wild images where subtle expressions matter.
* **Role in VOIX:** Provides the state-of-the-art 3D reconstruction backbone for more accurate separation of identity shape from emotional expression on unconstrained video data, ensuring VOIX craniofacial prior is not contaminated by transient expressions.

---

## BUCKET 3: Speaker Representation Learning, Neural Codecs and Zero-Shot TTS
> **Research Goal:** Ground the target acoustic embedding space, discrete neural audio codecs, and state-of-the-art zero-shot neural vocoding.
> **Mapped Module:** Module 2: Acoustic Representation and Module 4: Speech Synthesis

* **Core Research Question:** What representation space isolates vocal timbre and tract resonance, and how can an externally predicted speaker vector drive zero-shot waveform generation?

### Paper 9: ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification
* **Authors:** Brecht Desplanques, Jenthe Thienpondt, Kris Demuynck
* **Venue:** Interspeech, 2020
* **Era / Category:** Standard Acoustic Backbone
* **Scope and Methodology:** Integrates 1D dilated Squeeze-and-Excitation channel attention blocks into Time-Delay Neural Networks with multi-scale feature propagation. Aggregates global temporal context via Attentive Statistics Pooling into a 192-d speaker identity embedding (x-vector).
* **Role in VOIX:** The ECAPA-TDNN 192-d speaker embedding is VOIX ground-truth acoustic representation space. The CVAE face adapter is trained to predict distributions over this embedding space from craniofacial features, making ECAPA-TDNN embedding space the acoustic target manifold.

### Paper 10: High-Fidelity Audio Compression with Improved RVQGAN (Descript Audio Codec)
* **Authors:** Rithesh Kumar, Prem Seetharaman, Alejandro Luebs, Ishaan Kumar, Kundan Kumar
* **Venue:** Advances in Neural Information Processing Systems (NeurIPS), 36, 2023, pp. 27980-27993
* **Era / Category:** Bleeding-Edge Neural Codec (2023)
* **Scope and Methodology:** Universal discrete neural audio codec using Residual Vector Quantization with improved adversarial training. Compresses 44.1 kHz audio to 8 kbps (~90x compression) into compact discrete token sequences with minimal perceptual quality loss.
* **Role in VOIX:** Provides high-fidelity acoustic tokenization as an alternative representation pathway for discrete generative modeling. Informs VOIX evaluation of whether to target continuous speaker embeddings (ECAPA) vs. discrete acoustic tokens (RVQ).

### Paper 11: StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models
* **Authors:** Yinghao Aaron Li, Cong Han, Vinay S. Raghavan, Gavin Mischler, Nima Mesgarani
* **Venue:** Advances in Neural Information Processing Systems (NeurIPS), 36, 2023, pp. 19594-19621
* **Era / Category:** Modern Speech Synthesis SOTA (2023)
* **Scope and Methodology:** Non-autoregressive zero-shot TTS combining style diffusion (sampling style vectors from a diffusion prior), adversarial training with large speech discriminators, and differentiable duration modeling. Achieves human-level naturalness on LJSpeech conditioned only on a reference speaker embedding.
* **Role in VOIX:** Acts as the VOIX speech synthesis backend. Takes the CVAE-predicted speaker embedding and an input text transcript to synthesize a natural 24 kHz waveform in the predicted voice without requiring enrollment speech from the target speaker.

### Paper 12: NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models
* **Authors:** Zeqian Ju, Yuancheng Wang, Kai Shen, Xu Tan, Detai Xin, Dongchao Yang, and others
* **Venue:** arXiv preprint arXiv:2403.03100, 2024
* **Era / Category:** Bleeding-Edge Speech SOTA (2024)
* **Scope and Methodology:** Factorizes speech into disentangled subspaces - content, prosody, acoustic timbre, and fine acoustic details - using a codec with factorized vector quantization, then applies a separate diffusion model to each subspace. Achieves zero-shot synthesis from a 3-second reference clip.
* **Role in VOIX:** Validates the scientific principle that timbre must be factorized and isolated from prosody and content before it can be conditioned on biometric priors. Justifies VOIX design decision to operate on the ECAPA-TDNN timbre embedding rather than raw waveforms.

---

## BUCKET 4: Probabilistic Generative Modeling: CVAE to Flow Matching and Latent Diffusion
> **Research Goal:** Mathematically justify and formulate conditional latent distributions to sample diverse, plausible voices for a single face without mode collapse.
> **Mapped Module:** Module 3: Probabilistic Face-to-Voice CVAE Adapter

* **Core Research Question:** How do we model ill-posed, ambiguous one-to-many cross-modal distributions without posterior collapse?

### Paper 13: Learning Structured Output Representation Using Deep Conditional Generative Models (CVAE)
* **Authors:** Kihyuk Sohn, Honglak Lee, Xinchen Yan
* **Venue:** Advances in Neural Information Processing Systems (NeurIPS), 28, 2015
* **Era / Category:** Foundational Mathematical Pioneer
* **Scope and Methodology:** Formulates the Conditional Variational Autoencoder (CVAE) and the conditional ELBO training objective. Trains a recognition network q(z|x,y) to approximate the intractable posterior and a generative network p(y|x,z) to reconstruct the output conditioned on input and latent.
* **Role in VOIX:** Provides the core mathematical framework for the VOIX probabilistic adapter. Visual craniofacial features form the conditioning variable x, the acoustic embedding is y, and z captures the residual aleatoric uncertainty (soft tissue, habitual prosody) not visible from the face.

### Paper 14: beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework
* **Authors:** Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, Alexander Lerchner
* **Venue:** International Conference on Learning Representations (ICLR), 2017
* **Era / Category:** Foundational Disentanglement
* **Scope and Methodology:** Introduces a scalar hyperparameter beta > 1 weighting the KL-divergence term in the ELBO, acting as an information bottleneck. Higher beta encourages disentangled, interpretable latent representations at the cost of reconstruction fidelity.
* **Role in VOIX:** Supplies the theoretical basis for the dynamic beta-annealing schedule used during CVAE adapter training to prevent posterior collapse (where the decoder ignores the latent noise z and the model degenerates into a deterministic map).

### Paper 15: Flow Matching for Generative Modeling
* **Authors:** Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le
* **Venue:** arXiv preprint arXiv:2210.02747, 2022
* **Era / Category:** Modern Generative SOTA (2022-2024)
* **Scope and Methodology:** Formulates Continuous Normalizing Flows via optimal-transport flow matching, learning a vector field that maps noise to data along straight-line trajectories. Avoids the slow iterative sampling and instabilities of traditional diffusion models and Gaussian VAEs.
* **Role in VOIX:** Provides the modern architectural alternative to CVAE sampling. Evaluated as a potential upgrade path in VOIX ablation studies: comparing CVAE adapter vs. flow-matching adapter vs. latent diffusion for the face-to-embedding prediction task.

### Paper 16: High-Resolution Image Synthesis with Latent Diffusion Models
* **Authors:** Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Bjorn Ommer
* **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 10674-10685
* **Era / Category:** Modern Latent Architecture (2022)
* **Scope and Methodology:** Moves diffusion processes into a compact latent space encoded by a pretrained VQ-regularized autoencoder, dramatically reducing computational cost while preserving perceptual quality. Introduces cross-attention conditioning for text and other modalities.
* **Role in VOIX:** Validates VOIX foundational design decision: predict a compact 192-d speaker embedding latent rather than attempting to generate raw waveforms directly from face pixels. Generative processes in low-dimensional latent spaces are more tractable, stable, and controllable.

---

## BUCKET 5: Craniofacial Biomechanics, Formants and Physical Bounds
> **Research Goal:** Ground the system in vocal tract physical acoustics and establish the theoretical and empirical ceiling of face-voice correlation.
> **Mapped Module:** Module 5: Forensic Biomechanics and Biological Priors

* **Core Research Question:** What are the physical relationships between skull dimensions and vocal acoustics, and what is the biological limit of mutual predictability?

### Paper 17: Morphology and Development of the Human Vocal Tract: A Study Using Magnetic Resonance Imaging
* **Authors:** W. Tecumseh Fitch, Jay Giedd
* **Venue:** The Journal of the Acoustical Society of America (JASA), 106(3), 1999, pp. 1511-1522
* **Era / Category:** Foundational Acoustic Morphology
* **Scope and Methodology:** Volumetric MRI measurements of 129 normal humans aged 2-25 years. Quantifies midsagittal vocal tract length, shape, and oral/pharyngeal cavity proportions. Finds significant positive correlation between vocal tract length and body size (height/weight), and documents the sex-differentiated vocal tract remodeling arising at puberty.
* **Role in VOIX:** Provides the foundational normative MRI data establishing that skull/body dimensions physically constrain vocal tract geometry. Proves that craniofacial morphology captured by FLAME shape parameters encodes biomechanically meaningful priors for voice production, justifying the core VOIX hypothesis.

### Paper 18: Correlation Between the Length and Sagittal Projection of the Upper and Lower Jaw and the Fundamental Frequency
* **Authors:** Antoine T. Macari, Iyad A. Karam, Dany Tabri, Diana Sarieddine, Abdul-Latif Hamdan
* **Venue:** Journal of Voice, 28(3), 2014, pp. 291-296
* **Era / Category:** Empirical Acoustic Biology
* **Scope and Methodology:** Prospective study of 45 healthy subjects using cephalometric facial skeletal measurements (SNA, SNB, ANB angles; mandible length Co-Gn; maxilla length PNS-ANS) correlated against fundamental frequency F0 via acoustic analysis (VISI-PITCH IV). Found moderate negative correlation between jaw length and F0 (r approximately -0.53 to -0.58), with poor correlation for remaining cephalometric parameters.
* **Role in VOIX:** Provides direct empirical evidence that craniofacial skeletal dimensions correlate with F0 but only moderately, scientifically defending VOIX use of probabilistic modeling: the weak-to-moderate correlations mean deterministic regression is biologically inappropriate and aleatoric uncertainty modeling is mandated.

### Paper 19: Rethinking Voice-Face Correlation: A Geometry View
* **Authors:** Xiang Li, Yandong Wen, Muqiao Yang, Jinglu Wang, Rita Singh, Bhiksha Raj
* **Venue:** Proceedings of the 31st ACM International Conference on Multimedia (ACM MM), 2023, pp. 2458-2467
* **Era / Category:** Modern Geometric Cross-Modal Analysis (2023)
* **Scope and Methodology:** Proposes a Voice-Anthropometric Measurement (AM)-Face paradigm: directly predicts precise facial anthropometric measurements (distances and proportions between 3D facial landmarks) from acoustic features, then uses these predicted AMs to guide 3D face reconstruction. Demonstrates that nasal cavity and cranial geometry show the strongest geometric correlations with voice acoustics.
* **Role in VOIX:** Serves as the modern SOTA biological and geometric justification for VOIX approach, empirically confirming that specific 3D cranial skeletal landmarks are predictable from voice without relying on superficial semantic or demographic shortcuts. Directly validates VOIX use of FLAME shape parameters as the acoustic target correlate.

---

## BUCKET 6: Cross-Modal Identity, Audiovisual Corpora, Accent Fairness and Audio Forensics
> **Research Goal:** Ground dataset curation, cross-modal identity retrieval protocols, cross-demographic evaluation (Indian cohort), and state-of-the-art synthetic audio watermarking.
> **Mapped Module:** Module 6: Datasets, Module 7: Evaluation Framework and Module 8: Ethical Guardrails

* **Core Research Question:** What benchmark standards exist for audiovisual identity, how do cross-modal retrieval systems perform across demographics, and what guardrails prevent deepfake abuse?

### Paper 20: Learnable PINs: Cross-Modal Embeddings for Person Identity
* **Authors:** Arsha Nagrani, Samuel Albanie, Andrew Zisserman
* **Venue:** European Conference on Computer Vision (ECCV), 2018, pp. 73-89
* **Era / Category:** Cross-Modal Identity Benchmark
* **Scope and Methodology:** Proposes learning identity-sensitive joint face-voice embeddings without identity labels, using cross-modal self-supervision from unlabelled video. Introduces curriculum hard-negative mining for cross-modal retrieval. Evaluates on identities unseen during training, establishing a benchmark for the novel face-voice cross-modal retrieval task.
* **Role in VOIX:** Provides VOIX primary unsupervised cross-modal evaluation framework. The Learnable PINs retrieval benchmark (face to voice and voice to face) serves as the core protocol for measuring whether VOIX-generated voices are identity-consistent with the conditioning face.

### Paper 21: VoxCeleb2: Deep Speaker Recognition
* **Authors:** Joon Son Chung, Arsha Nagrani, Andrew Zisserman
* **Venue:** arXiv preprint arXiv:1806.05622, 2018; Interspeech, 2018
* **Era / Category:** Benchmark Dataset Standard
* **Scope and Methodology:** Large-scale audio-visual speaker recognition dataset collected via automated pipeline from YouTube interviews. Contains over 1 million utterances from 6,112 celebrity identities with accompanying face tracks, with no studio conditions - fully in-the-wild.
* **Role in VOIX:** Primary training dataset for the VOIX Face Adapter. The paired face tracks and speaker audio provide the ground-truth (craniofacial features, ECAPA-TDNN embedding) pairs used for CVAE adapter training.

### Paper 22: Looking to Listen at the Cocktail Party: A Speaker-Independent Audio-Visual Model for Speech Separation
* **Authors:** Ariel Ephrat, Inbar Mosseri, Oran Lang, Tali Dekel, Kevin Wilson, Avinatan Hassidim, William T. Freeman, Michael Rubinstein
* **Venue:** arXiv preprint arXiv:1804.03619, 2018; ACM SIGGRAPH, 2018
* **Era / Category:** Audio-Visual Benchmark Foundation
* **Scope and Methodology:** Speaker-independent audio-visual speech separation model. Introduces the AVSpeech dataset: a large-scale audiovisual dataset of millions of clean speech segments with corresponding face tracks, each verified to contain a single active speaker with high SNR.
* **Role in VOIX:** The AVSpeech dataset quality filtering criteria (active speaker verification, high SNR, single speaker per clip) inform VOIX data curation protocol for constructing the training splits from VoxCeleb2.

### Paper 23: Demographic Fairness in Multimodal Biometrics: A Comparative Analysis on Audio-Visual Speaker Recognition Systems
* **Authors:** Gianni Fenu, Mirko Marras
* **Venue:** Procedia Computer Science, 198, 2022, pp. 249-254
* **Era / Category:** Bias and Fairness Analysis (2022)
* **Scope and Methodology:** Comparative analysis of demographic disparities (gender, nationality, age) in multimodal audio-visual speaker recognition systems. Quantifies differential error rates across demographic groups, finding significant performance degradation for underrepresented demographic cohorts.
* **Role in VOIX:** Provides scientific grounding for VOIX cross-demographic evaluation design. Justifies construction of a dedicated 50-speaker Indian Audio-Visual Evaluation Cohort, demonstrating that Western-trained models exhibit measurable bias against non-Western speakers.

### Paper 24: SVARAH: Evaluating English ASR Systems on Indian Accents
* **Authors:** Tahir Javed, Sakshi Joshi, Vignesh Nagarajan, Sai Sundaresan, Janki Nawale, Ankur Raman, Kaushal Bhogale, Pratyush Kumar, Mitesh M. Khapra
* **Venue:** arXiv preprint arXiv:2305.15760, 2023
* **Era / Category:** Indian Accent Evaluation (2023)
* **Scope and Methodology:** Constructs a benchmark (SVARAH) for evaluating ASR systems on Indian English accents, covering 117 speakers across 19 Indian states with diverse mother tongues. Documents significant performance degradation of English ASR systems on Indian accents relative to native English.
* **Role in VOIX:** Provides domain-specific evidence of the Indian accent distribution shift problem. Directly motivates VOIX dedicated Indian cohort evaluation, demonstrating that accent-related acoustic feature shifts are sufficiently large to require explicit out-of-distribution evaluation.

### Paper 25: Proactive Detection of Voice Cloning with Localized Watermarking (AudioSeal)
* **Authors:** Robin San Roman, Pierre Fernandez, Alexandre Defossez, Teddy Furon, Tuan Tran, Hady Elsahar
* **Venue:** arXiv preprint arXiv:2401.17264, 2024
* **Era / Category:** Bleeding-Edge Audio Forensics (2024 SOTA)
* **Scope and Methodology:** Neural audio watermarking system (AudioSeal) that embeds imperceptible, sample-level localized watermarks into generated speech during synthesis. A detector network localizes watermarked segments with high accuracy even after compression, noise, and audio editing attacks. Operates at real-time speed.
* **Role in VOIX:** Directly integrated into Module 8 (Ethical and Forensic Safeguards). All VOIX-synthesized audio is watermarked using AudioSeal during generation, ensuring that all outputs are cryptographically attributable to VOIX and detectable as synthetic, preventing deepfake misuse.

---

## Lit Survey Execution Roadmap

```
Phase 1: Feature Representation and Encoders (Buckets 1, 2, 3)
+-- Establish cross-modal baselines: Speech2Face (P1), Seeing Voices (P2), Imaginary Voice (P4).
+-- Integrate ArcFace 512-d (P5) + FLAME 3DMM shape via DECA (P7) + EMOCA (P8).
+-- Target ECAPA-TDNN 192-d speaker embeddings (P9); compare DAC neural tokens (P10).
    +-- Synthesis backend: StyleTTS 2 (P11), informed by NaturalSpeech 3 factorization (P12).

Phase 2: Generative Formulations and Physical Bounds (Buckets 4, 5)
+-- Implement CVAE conditional ELBO (P13) with beta-annealing schedule (P14).
+-- Ablate against Flow Matching (P15) and Latent Diffusion (P16) adapter variants.
+-- Constrain physical plausibility with Fitch and Giedd VTL bounds (P17),
    Macari jaw-F0 correlations (P18), and Li et al. anthropometric geometry (P19).

Phase 3: Benchmarking, Validation and Safeguards (Bucket 6)
+-- Primary cross-modal evaluation: Learnable PINs protocol (P20).
+-- Training data: VoxCeleb2 (P21), curated via AVSpeech quality criteria (P22).
+-- Demographic fairness audit (P23); Indian accent evaluation via SVARAH-informed cohort (P24).
+-- Ethical safeguard: AudioSeal watermarking (P25) on all synthesized outputs.
```
