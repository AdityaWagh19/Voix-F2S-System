"""Neural network model definitions and voice transformation engines."""

from voix.models.adapter import StyleAdapter
from voix.models.cvae import CVAE, CVAEDecoder, CVAEEncoder, CVAEOutput
from voix.models.tts_backend import StyleTTS2Backend
from voix.models.watermarking import AudioSealWatermarker
from voix.models.speech_simulation import (
    SpeechDifferenceType,
    SpeechSimulator,
    SpeechTextTransformer,
    SpeechAcousticTransformer,
)
from voix.models.accent_control import (
    AccentType,
    AccentProfile,
    AccentTransformer,
    list_available_accents,
    ACCENT_REGISTRY,
)
from voix.models.cross_lingual import (
    LanguageType,
    LanguageProfile,
    MultilingualPhoneticAdapter,
    list_supported_languages,
    LANGUAGE_REGISTRY,
)
from voix.models.emotion_control import (
    EmotionType,
    EmotionProfile,
    EmotionModulator,
    list_available_emotions,
    EMOTION_REGISTRY,
)

__all__ = [
    "CVAE",
    "CVAEEncoder",
    "CVAEDecoder",
    "CVAEOutput",
    "StyleAdapter",
    "StyleTTS2Backend",
    "AudioSealWatermarker",
    "SpeechDifferenceType",
    "SpeechSimulator",
    "SpeechTextTransformer",
    "SpeechAcousticTransformer",
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
    "LanguageType",
    "LanguageProfile",
    "MultilingualPhoneticAdapter",
    "list_supported_languages",
    "LANGUAGE_REGISTRY",
]
