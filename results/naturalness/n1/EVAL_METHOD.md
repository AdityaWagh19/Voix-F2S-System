# N1 evaluation method (low-RAM split)

The N1 pilot is scored on all 104 held-out AMI validation clips (4 speakers).
The pre-registered gate is unchanged:

- WER within `base + 0.05`;
- target and reference ECAPA similarity within `base - 0.05`;
- at least one prosody/naturalness proxy closer to the real AMI target with a
  speaker-cluster bootstrap 95% CI whose upper bound is below zero;
- all 104 base/LoRA pairs must be present. A partial merge cannot pass the
  gate, even if its subset looks favorable.

## Why the evaluation is split

This machine has 7.8 GB RAM and roughly 2 GB free. Holding StyleTTS2, ECAPA,
librosa, and Whisper in one process exhausted the pagefile and crashed after a
few clips. The evaluation therefore runs in three stages, each in a fresh
process:

1. paired base/LoRA generation (`--eval-generate-only`);
2. bounded metric slices (`evaluate_n1_metrics.py`, 8 rows per process);
3. merge (`merge_n1_eval.py`) and gate scoring (`evaluate_n1_gate.py`).

This changes process boundaries, not the clips, seeds, model, or gate.

## Commands

```powershell
$env:TMP='D:\voix\.tmp'; $env:TEMP='D:\voix\.tmp'
$env:HF_HOME='D:\voix\.cache\huggingface'; $env:TORCH_HOME='D:\voix\.cache\torch'
$env:XDG_CACHE_HOME='D:\voix\.cache'; $env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'; $env:NUMEXPR_NUM_THREADS='1'

python scripts\train_styletts2_lora.py `
  --manifest data\processed\ami_lora\manifest.csv `
  --output results\naturalness\n1\ami_lora_pilot.pt `
  --eval-only --eval-generate-only --eval-samples 104 `
  --eval-dir results\naturalness\n1\eval_pilot `
  --eval-output results\naturalness\n1\eval_pilot.json `
  --device cuda --eval-device cpu --seed 42
```

```powershell
$parts = @()
for ($i = 0; $i -lt 208; $i += 8) {
  $out = "results\naturalness\n1\parts\part_{0:D3}.json" -f [int]($i / 8)
  if (-not (Test-Path $out)) {
    python scripts\evaluate_n1_metrics.py `
      --eval results\naturalness\n1\eval_pilot.json `
      --offset $i --limit 8 --output $out --device cpu
    if ($LASTEXITCODE -ne 0) { break }
  }
  $parts += $out
}
python scripts\merge_n1_eval.py --parts $parts `
  --output results\naturalness\n1\eval_pilot_merged.json
python scripts\evaluate_n1_gate.py `
  --eval results\naturalness\n1\eval_pilot_merged.json `
  --output results\naturalness\n1\gate_report.json
```

## Scope and non-claims

- 104 clips from 4 held-out speakers; this is a pilot, not a confirmatory
  naturalness study.
- The adapter is LoRA rank 8 on `model.diffusion.unet` only; the predictor is
  not adapted.
- The training run was capped at 400 steps (one epoch).
- No MOS, CMOS, or listener study was run. A gate pass is a proxy result.
