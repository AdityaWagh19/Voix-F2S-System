"""Dynamic Cross-Platform Hardware Auto-Configuration Engine for VOIX.

Detects host operating system (Windows, Linux, macOS) and hardware accelerators:
  - NVIDIA CUDA (Tier 1 Budget <=4GB, Tier 2 Mid 6-12GB, Tier 3 High >=16GB)
  - AMD ROCm (via PyTorch HIP runtime)
  - Apple Silicon MPS (M1/M2/M3/M4 unified memory on macOS)
  - Intel XPU (Intel Extension for PyTorch)
  - Universal Multi-Core CPU (x86_64, aarch64)

Resolves optimal runtime profiles, precision, memory staging, and diffusion steps
to maximize inference velocity without compromising 24 kHz acoustic synthesis fidelity.
"""

from __future__ import annotations

import gc
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import yaml


@dataclass
class HardwareProfile:
    """Resolved runtime configuration for active hardware."""
    tier_id: str
    name: str
    os_name: str
    arch: str
    accelerator: str                       # 'cuda', 'mps', 'xpu', 'cpu'
    device: torch.device
    total_memory_gb: float
    pipeline_mode: str                     # 'staged_memory_isolated', 'concurrent_gpu', 'batched_gpu', 'multithreaded_cpu'
    precision: str                         # 'fp16', 'fp32'
    tts_device: str                        # Target device for StyleTTS 2 vocoder
    diffusion_steps: int                   # Neural vocoder sampling steps
    batch_voices: bool                     # Whether to batch K candidate voices
    torch_empty_cache_per_sample: bool     # Proactive cache flush
    onnx_execution_provider: str          # 'CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider'
    cpu_threads: int
    raw_config: Dict[str, Any] = field(default_factory=dict)

    def print_summary(self) -> None:
        """Print concise diagnostic summary of active hardware profile."""
        print(f"[VOIX Hardware] OS: {self.os_name} ({self.arch}) | Device: {self.device}")
        print(f"[VOIX Hardware] Profile: '{self.name}' (Tier: {self.tier_id})")
        print(f"[VOIX Hardware] Memory: {self.total_memory_gb:.2f} GB | Pipeline Mode: {self.pipeline_mode}")
        print(f"[VOIX Hardware] Vocoder Device: {self.tts_device} | Steps: {self.diffusion_steps} | Precision: {self.precision}")


def get_system_ram_gb() -> float:
    """Get total host physical RAM in gigabytes across Windows, Linux, and macOS."""
    try:
        import psutil
        return float(psutil.virtual_memory().total) / 1e9
    except ImportError:
        pass

    # Platform-specific native fallbacks
    sys_name = platform.system()
    if sys_name == "Windows":
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return float(stat.ullTotalPhys) / 1e9
        except Exception:
            return 8.0
    elif sys_name == "Darwin":
        try:
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            return float(out) / 1e9
        except Exception:
            return 8.0
    else:  # Linux
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = float(line.split()[1])
                        return kb / 1e6
        except Exception:
            return 8.0

    return 8.0


def free_memory(device: Optional[torch.device] = None) -> None:
    """Proactively trigger garbage collection and flush accelerator cache."""
    gc.collect()
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device is not None and device.type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def detect_hardware(config_path: Optional[Union[str, Path]] = None) -> HardwareProfile:
    """Automatically detect host OS and accelerator, and resolve optimal runtime profile.

    Args:
        config_path: Optional path to configs/hardware.yaml. If None or missing,
                     dynamic fallbacks are used.

    Returns:
        HardwareProfile: Fully resolved configuration object.
    """
    os_name = platform.system()
    arch = platform.machine()
    system_ram_gb = get_system_ram_gb()

    # 1. Determine Accelerator
    accelerator = "cpu"
    vram_gb = 0.0
    device = torch.device("cpu")

    # Check CUDA (NVIDIA or AMD ROCm via HIP)
    if torch.cuda.is_available():
        accelerator = "cuda"
        device = torch.device("cuda:0")
        try:
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = float(vram_bytes) / 1e9
        except Exception:
            vram_gb = 4.0
    # Check Apple Silicon MPS
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        accelerator = "mps"
        device = torch.device("mps")
        vram_gb = system_ram_gb  # Unified memory architecture
    # Check Intel XPU
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        accelerator = "xpu"
        device = torch.device("xpu:0")
        try:
            vram_bytes = torch.xpu.get_device_properties(0).total_memory
            vram_gb = float(vram_bytes) / 1e9
        except Exception:
            vram_gb = 4.0
    else:
        accelerator = "cpu"
        device = torch.device("cpu")
        vram_gb = system_ram_gb

    # 2. Load configurations or defaults
    cfg: Dict[str, Any] = {}
    if config_path is not None and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

    thresholds = cfg.get("thresholds", {
        "budget_vram_limit_gb": 4.5,
        "mid_vram_limit_gb": 12.5,
        "high_vram_limit_gb": 16.0,
    })
    profiles = cfg.get("profiles", {})

    # 3. Classify Tier
    tier_id = "tier_cpu_only"
    if accelerator == "cuda":
        if vram_gb <= thresholds.get("budget_vram_limit_gb", 4.5):
            tier_id = "tier_1_budget_gpu"
        elif vram_gb <= thresholds.get("mid_vram_limit_gb", 12.5):
            tier_id = "tier_2_mid_gpu"
        else:
            tier_id = "tier_3_high_gpu"
    elif accelerator == "mps":
        tier_id = "tier_apple_silicon"
    else:
        tier_id = "tier_cpu_only"

    # 4. Resolve Profile Attributes
    p_data = profiles.get(tier_id, {})

    # Fallback attribute defaults if yaml was absent or incomplete
    default_defaults = {
        "tier_1_budget_gpu": {
            "name": "Entry Dedicated GPU (<= 4GB VRAM)",
            "pipeline_mode": "staged_memory_isolated",
            "precision": "fp16",
            "tts_device": "cuda",
            "diffusion_steps": 20,
            "batch_voices": False,
            "torch_empty_cache_per_sample": True,
            "onnx_execution_provider": "CPUExecutionProvider",
            "cpu_threads": 4,
        },
        "tier_2_mid_gpu": {
            "name": "Mainstream Dedicated GPU (6GB - 12GB VRAM)",
            "pipeline_mode": "concurrent_gpu",
            "precision": "fp16",
            "tts_device": "cuda",
            "diffusion_steps": 25,
            "batch_voices": True,
            "torch_empty_cache_per_sample": False,
            "onnx_execution_provider": "CUDAExecutionProvider",
            "cpu_threads": 6,
        },
        "tier_3_high_gpu": {
            "name": "Workstation / Datacenter GPU (>= 16GB VRAM)",
            "pipeline_mode": "batched_gpu",
            "precision": "fp16",
            "tts_device": "cuda",
            "diffusion_steps": 30,
            "batch_voices": True,
            "torch_empty_cache_per_sample": False,
            "onnx_execution_provider": "CUDAExecutionProvider",
            "cpu_threads": 8,
        },
        "tier_apple_silicon": {
            "name": "Apple Silicon Unified Memory (MPS)",
            "pipeline_mode": "concurrent_unified",
            "precision": "fp32",
            "tts_device": "mps",
            "diffusion_steps": 16,
            "batch_voices": False,
            "torch_empty_cache_per_sample": True,
            "onnx_execution_provider": "CoreMLExecutionProvider",
            "cpu_threads": 6,
        },
        "tier_cpu_only": {
            "name": "Universal CPU Engine",
            "pipeline_mode": "multithreaded_cpu",
            "precision": "fp32",
            "tts_device": "cpu",
            "diffusion_steps": 12,
            "batch_voices": False,
            "torch_empty_cache_per_sample": False,
            "onnx_execution_provider": "CPUExecutionProvider",
            "cpu_threads": max(1, (os.cpu_count() or 4) - 1),
        },
    }

    tier_defaults = default_defaults.get(tier_id, default_defaults["tier_cpu_only"])

    name = p_data.get("name", tier_defaults["name"])
    pipeline_mode = p_data.get("pipeline_mode", tier_defaults["pipeline_mode"])
    precision = p_data.get("precision", tier_defaults["precision"])
    tts_device = p_data.get("tts_device", tier_defaults["tts_device"])
    diffusion_steps = int(p_data.get("diffusion_steps", tier_defaults["diffusion_steps"]))
    batch_voices = bool(p_data.get("batch_voices", tier_defaults["batch_voices"]))
    torch_empty_cache = bool(p_data.get("torch_empty_cache_per_sample", tier_defaults["torch_empty_cache_per_sample"]))
    onnx_ep = p_data.get("onnx_execution_provider", tier_defaults["onnx_execution_provider"])
    
    cpu_threads_cfg = p_data.get("cpu_threads", tier_defaults["cpu_threads"])
    if cpu_threads_cfg == "auto" or not isinstance(cpu_threads_cfg, int):
        cpu_threads = max(1, (os.cpu_count() or 4) - 1)
    else:
        cpu_threads = int(cpu_threads_cfg)

    # Apply PyTorch runtime thread limits
    try:
        torch.set_num_threads(cpu_threads)
    except Exception:
        pass

    # Enable cuDNN benchmark for stable convolutional sizes
    if accelerator == "cuda":
        torch.backends.cudnn.benchmark = True

    return HardwareProfile(
        tier_id=tier_id,
        name=name,
        os_name=os_name,
        arch=arch,
        accelerator=accelerator,
        device=device,
        total_memory_gb=vram_gb if accelerator != "cpu" else system_ram_gb,
        pipeline_mode=pipeline_mode,
        precision=precision,
        tts_device=tts_device,
        diffusion_steps=diffusion_steps,
        batch_voices=batch_voices,
        torch_empty_cache_per_sample=torch_empty_cache,
        onnx_execution_provider=onnx_ep,
        cpu_threads=cpu_threads,
        raw_config=p_data,
    )
