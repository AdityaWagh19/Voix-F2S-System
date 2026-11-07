# N1 AMI LoRA smoke

> **Update (2026-11-02):** a 400-step N1 pilot has since trained and its
> pre-registered proxy gate passed narrowly on all 104 validation pairs. The
> authoritative result is [`../FINDINGS.md`](../FINDINGS.md). This smoke is
> retained as the low-VRAM pipeline record; it is not the N1 result.

**Status: archived pipeline smoke; no naturalness claim from the four-step run.**
The AMI LoRA trainer loaded StyleTTS2, injected LoRA, ran four optimizer steps
and a full validation pass with zero failures. This proves the manifest,
tokenizer, reference cache, LoRA injection, optimizer, and validation path work
on real AMI audio. It does not test whether adaptation improves naturalness:
generation evaluation was disabled in the low-VRAM mode, and no WER, prosody,
identity, or MOS comparison was run for the smoke.

## What ran

- Manifest: `data/processed/ami_lora/manifest.csv` (2,424 rows).
  - train: 2,320 clips / 8 speakers / 2.79 h
  - validation: 104 clips / 4 speakers / 0.13 h
  - human AMI transcripts, deterministic same-speaker reference clips,
    `gender="unknown"`.
  - AMI headset audio is 16 kHz and noisy; the loader resamples to 24 kHz for
    StyleTTS2.
- Command:

  ```text
  python scripts/train_styletts2_lora.py \
    --manifest data/processed/ami_lora/manifest.csv \
    --output results/naturalness/n1/ami_lora_smoke.pt \
    --history results/naturalness/n1/ami_lora_smoke.history.json \
    --max-steps 4 --epochs 1 --val-repeats 1 --eval-samples 0 \
    --device cuda --load-on-cpu
  ```

- Low-VRAM mode (`--load-on-cpu`): StyleTTS2 is constructed on CPU; only BERT,
  the style/predictor encoders, and the diffusion U-Net are moved to the GPU.
  The text aligner, F0 extractor, decoder, and vocoder are released. This mode
  is training-only and refuses `--eval-samples > 0`.

## Results

| Item | Value |
|---|---|
| Optimizer steps | 4 |
| Train clips seen | 16 (batch 1, grad accumulation 4) |
| Validation clips | 104 (all validation clips) |
| Failures | 0 |
| LoRA layers | 25 |
| Trainable parameters | 434,184 |
| Train loss | 0.953079 |
| Validation loss | 0.798519 |
| Learning rate | 2e-4 configured; 1e-5 recorded after step 4 (cosine floor) |
| Model load time | 64.1 s |
| Training + validation loop | 34.1 s |
| Checkpoint size | 1.76 MB |

The loss values are from four steps and are not evidence of convergence,
overfitting, or a naturalness effect. The checkpoint stores only the LoRA
adapter weights.

## Deviations and limits

- **Diffusion-only adapter.** The trainer injects LoRA into
  `model.diffusion.unet` only. The plan specifies `model.predictor` and
  `model.diffusion`. A predictor adapter is not implemented; this smoke is
  narrower than the pre-registered N1 intervention.
- **Below the N1 speaker target.** 8 training speakers and 4 validation
  speakers, versus the 10 held-out speaker target. This is a pilot, not
  confirmatory.
- **No generation evaluation.** Low-VRAM mode cannot synthesize audio, so WER,
  prosody, identity, and MOS are all `unavailable`.
- **No test split touched.** The 4 test speakers remain held out.

## Resource blocker

The 4 GB GTX 1650 cannot hold the full StyleTTS2 stack at once. The first smoke
OOM'd during model loading (`torch.OutOfMemoryError`, 20 MB allocation, 3.22 GiB
reported free). A standalone allocation test showed 2.0 GB succeeds but 2.5 GB
fails, consistent with a WDDM per-process allocation limit. A retry then failed
at `import torch` with `OSError: [WinError 1455] The paging file is too small`
when the C: drive had 0.25 GB free. The successful run used the low-VRAM path
after free space recovered to ~4.9 GB.

This is a host limitation, not a code failure. A full N1 evaluation needs
either more GPU memory or a CPU generation path; neither was available here.

## Smoke decision (historical)

- The four-step smoke alone is not an N1 naturalness result and must not be
  cited as one.
- The later 400-step pilot used generation-capable evaluation and passed the
  pre-registered proxy gate narrowly; see [`../FINDINGS.md`](../FINDINGS.md).
  That pass is also not a MOS/CMOS result.
- N2 is still not entered. Confirmatory work needs more held-out speakers,
  multiple seeds, the full predictor+diffusion adapter, and human listening.
- Keep the diffusion-only deviation and the 8-speaker pilot size in any future
  N1 report.

## Artifacts

- `results/naturalness/n1/ami_lora_smoke.pt` - LoRA adapter checkpoint.
- `results/naturalness/n1/ami_lora_smoke.history.json` - loss history.
- `results/naturalness/n1/ami_lora_smoke_lowvram.log` - full run log.
- `results/naturalness/n1/ami_lora_smoke.log` - first, failed full-GPU attempt.
