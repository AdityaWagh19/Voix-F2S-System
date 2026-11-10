# Dependency constraints

`serving-py312-cu121.txt` pins the direct dependency stack validated on the
audited Windows 11 / Python 3.12 / CUDA 12.1 workstation. It exists because the
old `requirements.txt` was a UTF-16 `pip freeze` of a polluted user site: it
pinned incompatible versions and omitted the serving packages entirely.

## Use

```powershell
python scripts/setup/bootstrap_env.py --recreate
```

or manually:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.4.1 torchaudio==2.4.1 `
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[serving]" -c constraints/serving-py312-cu121.txt
python -m pip check
```

For CPU-only or non-Windows machines, omit the CUDA index and the constraints
file; let pip resolve the platform wheels:

```bash
python -m pip install -e ".[serving]"
```

## Scope and limits

- These are version pins, not a hash-locked wheelhouse. A future transitive
  release can still change the resolved graph.
- `torch==2.4.1+cu121` only installs from the PyTorch CUDA 12.1 index. The
  bootstrap script and the commands above select that index explicitly.
- Regenerate the pin list after a deliberate upgrade by rerunning the install
  in a clean venv, recording `python -m pip freeze`, and re-running
  `python -m pytest` plus the CLI smoke tests. Do not copy a freeze from an
  environment with `include-system-site-packages = true`.
- The evaluation extra is optional and is not required for synthesis.
