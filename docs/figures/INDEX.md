# VOIX Figures Index
> Publication-quality methodology diagrams, architecture visualizations, and result graphs.
> All quantitative figures use verified data from 
esults/benchmarks/summary.csv.
> Generated: 2026-10-13 by docs/figures/gen_figures.py.

---

## Figure Inventory

| File | Title | Type | Data Source |
|---|---|---|---|
| [fig1_ablation_overview.png](fig1_ablation_overview.png) | 4-Panel Paradigm Ablation Overview | Quantitative results | summary.csv (verified) |
| [fig2_gender_secs.png](fig2_gender_secs.png) | Gender Accuracy vs SECS Dual-Axis | Quantitative results | summary.csv (verified) |
| [fig3_paradigm_progression.png](fig3_paradigm_progression.png) | Paradigm Development Trajectory | Quantitative results | summary.csv (verified) |
| [fig4_feature_composition.png](fig4_feature_composition.png) | 560-D Feature Composition & Demographics Layout | Architecture diagram | Source code verified |
| [fig5_hardware_profiles.png](fig5_hardware_profiles.png) | Hardware Tier VRAM Profiles | Architecture diagram | configs/hardware.yaml |
| [fig6_recall_k.png](fig6_recall_k.png) | Zero-Shot Recall@K All Models | Quantitative results | summary.csv (verified) |
| [fig7_cvae_architecture.png](fig7_cvae_architecture.png) | CVAE v2 Architecture Diagram | Architecture diagram | voix/models/cvae.py |
| [fig8_biomechanical_coupling.png](fig8_biomechanical_coupling.png) | Craniofacial-Vocal Coupling Hypotheses | Conceptual (HYPOTHESIZED) | Literature + voix/data/morphology.py |
| [fig9_style_adapter.png](fig9_style_adapter.png) | StyleAdapter + VocoderStyleBridge Pipeline | Architecture diagram | voix/models/adapter.py, tts_backend.py |
| [fig10_normalized_comparison.png](fig10_normalized_comparison.png) | Normalized Multi-Metric Comparison | Quantitative results | summary.csv (verified) |

---

## Usage in Papers

`latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=\linewidth]{figs/fig1_ablation_overview}
  \caption{VOIX paradigm ablation across five model architectures evaluated on 400 zero-shot
  validation identities. Val Cosine measures prediction alignment; SECS measures
  synthesized audio speaker similarity; Gender Acc measures demographic fidelity.
  Data from \texttt{results/benchmarks/summary.csv}.}
  \label{fig:ablation}
\end{figure}
`

---

## Status

| Metric | Coverage |
|---|---|
| Val Cosine | All 5 models (A, B, D, E, F) -- verified from CSV |
| Gender Accuracy | All 5 models -- verified from CSV |
| SECS Mean | All 5 models -- verified from CSV |
| Recall@1, @3, @10 | All 5 models -- verified from CSV |
| WER / FAD / MOS | NOT measured -- all -1.0 in CSV |
| Loss curves | ILLUSTRATIVE ONLY -- actual training logs not preserved |
| Biomechanical correlations | HYPOTHESIZED -- based on literature, not measured in this codebase |

---

## Regeneration

`ash
cd D:/voix
python docs/figures/gen_figures.py
`

Requires: matplotlib, numpy (already in project environment).
