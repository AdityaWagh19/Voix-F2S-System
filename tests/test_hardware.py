"""Unit tests for Dynamic Cross-Platform Hardware Auto-Configuration."""

import os
from unittest.mock import MagicMock, patch
import pytest
import torch

from voix.utils.hardware import (
    HardwareProfile,
    detect_hardware,
    free_memory,
    get_system_ram_gb,
)


class TestHardwareAutoDetection:
    """Verify hardware auto-configuration across platforms and accelerator tiers."""

    def test_live_detection_succeeds(self):
        """Ensure live environment detection produces valid non-null profile."""
        profile = detect_hardware()
        assert isinstance(profile, HardwareProfile)
        assert profile.os_name in ["Windows", "Linux", "Darwin"]
        assert profile.device is not None
        assert profile.diffusion_steps > 0
        assert profile.cpu_threads >= 1

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    def test_tier_1_budget_cuda_detection(self, mock_props, mock_cuda):
        """Verify <=4GB GPUs (e.g. GTX 1650) get mapped to Tier 1 staged isolated mode."""
        # Mock 4.0 GB VRAM
        mock_props.return_value = MagicMock(total_memory=4.0 * 1e9)

        profile = detect_hardware()
        assert profile.accelerator == "cuda"
        assert profile.tier_id == "tier_1_budget_gpu"
        assert profile.pipeline_mode == "staged_memory_isolated"
        assert profile.tts_device == "cuda"
        assert profile.diffusion_steps == 10
        assert profile.batch_voices is False

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    def test_tier_2_mid_cuda_detection(self, mock_props, mock_cuda):
        """Verify 6-12GB GPUs (e.g. RTX 4050/3060) get mapped to Tier 2 concurrent mode."""
        # Mock 8.0 GB VRAM
        mock_props.return_value = MagicMock(total_memory=8.0 * 1e9)

        profile = detect_hardware()
        assert profile.accelerator == "cuda"
        assert profile.tier_id == "tier_2_mid_gpu"
        assert profile.pipeline_mode == "concurrent_gpu"
        assert profile.tts_device == "cuda"
        assert profile.diffusion_steps == 15

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    def test_tier_3_high_cuda_detection(self, mock_props, mock_cuda):
        """Verify >=16GB GPUs (e.g. T4/A100) get mapped to Tier 3 batched parallel mode."""
        # Mock 24.0 GB VRAM
        mock_props.return_value = MagicMock(total_memory=24.0 * 1e9)

        profile = detect_hardware()
        assert profile.accelerator == "cuda"
        assert profile.tier_id == "tier_3_high_gpu"
        assert profile.pipeline_mode == "batched_gpu"
        assert profile.batch_voices is True
        assert profile.diffusion_steps == 20

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    @patch("platform.system", return_value="Darwin")
    def test_apple_silicon_mps_detection(self, mock_sys, mock_mps, mock_cuda):
        """Verify Apple Silicon M1/M2/M3 on macOS gets mapped to MPS profile."""
        profile = detect_hardware()
        assert profile.os_name == "Darwin"
        assert profile.accelerator == "mps"
        assert profile.tier_id == "tier_apple_silicon"
        assert profile.tts_device == "mps"
        assert profile.precision == "fp32"
        assert profile.diffusion_steps == 12

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    @patch("platform.system", return_value="Linux")
    def test_cpu_fallback_detection(self, mock_sys, mock_mps, mock_cuda):
        """Verify CPU-only environments on Linux get mapped to CPU tier with fast diffusion."""
        profile = detect_hardware()
        assert profile.os_name == "Linux"
        assert profile.accelerator == "cpu"
        assert profile.tier_id == "tier_cpu_only"
        assert profile.tts_device == "cpu"
        assert profile.diffusion_steps == 8

    def test_free_memory_callable(self):
        """Ensure memory clearing executes safely without error."""
        free_memory(torch.device("cpu"))
        if torch.cuda.is_available():
            free_memory(torch.device("cuda:0"))
