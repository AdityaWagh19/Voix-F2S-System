"""Unit test suite for VoixPipeline End-to-End Inference (Phase 4)."""

import os
import pytest
import soundfile as sf
import torch

from voix.inference.pipeline import VoixPipeline


class TestVoixPipeline:
    """Verify end-to-end multi-voice generation pipeline."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        return VoixPipeline(device="cpu", enable_watermarking=False)

    def test_pipeline_instantiation(self, pipeline):
        assert pipeline is not None
        assert pipeline.cvae is not None
        assert pipeline.adapter is not None
        assert pipeline.tts is not None

    def test_generate_from_random_face_vector(self, pipeline):
        face = torch.randn(560)
        results = pipeline.generate(
            face_input=face,
            text="Testing end-to-end voice generation.",
            num_voices=3,
            temperature=1.0,
            watermark=False,
        )

        assert len(results) == 3
        for idx, item in enumerate(results, 1):
            assert item["voice_index"] == idx
            assert item["sample_rate"] == 24000
            assert item["waveform"].dim() == 2
            assert item["waveform"].shape[0] == 1
            assert item["waveform"].shape[1] > 0
            assert item["speaker_embedding"].shape == (192,)
            assert item["style_embedding"].shape == (128,)

    def test_save_audio_files(self, pipeline, tmp_path):
        face = torch.randn(560)
        results = pipeline.generate(
            face_input=face,
            text="Short audio test.",
            num_voices=2,
            watermark=False,
        )

        saved = pipeline.save_audio(results, output_dir=str(tmp_path), prefix="test_voice")
        assert len(saved) == 2

        for path in saved:
            assert os.path.exists(path)
            data, sr = sf.read(path)
            assert sr == 24000
            assert len(data) > 0
    def test_generate_watermarked(self, pipeline):
        face = torch.randn(560)
        results = pipeline.generate(
            face_input=face,
            text="Watermarked audio test.",
            num_voices=1,
            watermark=True,
        )
        assert len(results) == 1
        assert results[0]["is_watermarked"] is True
        assert results[0]["waveform"].dim() == 2
