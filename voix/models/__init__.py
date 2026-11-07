"""Neural network model definitions and voice transformation engines."""

from voix.models.accent_control import (
    ACCENT_REGISTRY,
    AccentProfile,
    AccentTransformer,
    AccentType,
    list_available_accents,
)
from voix.models.adapter import StyleAdapter
from voix.models.cvae import CVAE, CVAEDecoder, CVAEEncoder, CVAEOutput
from voix.models.emotion_control import (
    EMOTION_REGISTRY,
    EmotionModulator,
    EmotionProfile,
    EmotionType,
    list_available_emotions,
)
from voix.models.tts_backend import StyleTTS2Backend
from voix.models.watermarking import AudioSealWatermarker

__all__ = [
    "CVAE",
    "CVAEEncoder",
    "CVAEDecoder",
    "CVAEOutput",
    "StyleAdapter",
    "StyleTTS2Backend",
    "AudioSealWatermarker",
    "AccentType",
    "AccentProfile",
    "AccentTransformer",
    "list_available_accents",
    "ACCENT_REGISTRY",
    "EmotionType",
    "EmotionProfile",
    "EmotionModulator",
    "list_available_emotions",
    "EMOTION_REGISTRY",
]
