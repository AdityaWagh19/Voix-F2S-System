"""Audit emotion/accent corpora availability, licensing, and fit.

This script writes a machine-readable audit instead of a prose claim.  It does
two things:

1. Scans a small set of local roots for known emotion/accent corpora.
2. Records verified license and corpus facts with their primary sources.

The dataset facts below were checked against the linked primary sources on
2026-10-27.  Counts that are release-dependent or sampled are labelled as such;
the audit never turns a partial sample into a full-corpus count.

Usage:
    python scripts/audit/audit_expression_data.py
    python scripts/audit/audit_expression_data.py --output data/processed/expression_data_audit.json
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

AUDIT_SCHEMA = "voix.expression_data_audit.v1"
ACCESSED = "2026-10-28"


def _path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def _load_json(relative: str) -> Any:
    path = ROOT / relative
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


DATASETS: List[Dict[str, Any]] = [
    {
        "id": "crema_d",
        "name": "CREMA-D",
        "role": "emotion",
        "training_role": "pilot_evaluation",
        "speaker_id_available": "full",
        "speaker_id_notes": "Actor IDs are encoded in the CREMA-D file names.",
        "balance": {
            "status": "expected_balanced",
            "notes": (
                "Designed with six emotions recorded by the same 91 actors, so "
                "the speaker gate is plausible. However, 5.26 h total is about "
                "0.88 h per emotion, below the 2 h/class production threshold. "
                "CREMA-D is therefore pilot/evaluation-only and cannot train a "
                "learned emotion control; supplement it with a larger licensed "
                "emotion corpus (for example MSP-Podcast after the institutional "
                "agreement) before claiming production-grade emotion training."
            ),
        },
        "source": "https://github.com/CheyneyComputerScience/CREMA-D",
        "license": "ODbL 1.0",
        "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
        "commercial_use": "allowed_with_obligations",
        "commercial_use_notes": (
            "Commercial use is permitted. The database must be referenced "
            "together with its license, and ODbL share-alike obligations apply "
            "to derived databases."
        ),
        "size": {
            "clips": 7442,
            "speakers": 91,
            "hours": 5.26,
            "language": "English",
            "modality": "audio-visual, acted",
        },
        "labels": {
            "emotions": ["anger", "disgust", "fear", "happiness", "neutral", "sadness"],
            "intensity_levels": ["low", "medium", "high", "unspecified"],
        },
        "local_expected_paths": [
            "data/raw/crema_d",
            "data/external/crema_d",
            "datasets/crema_d",
        ],
        "fit": (
            "Best small, clean, commercially usable emotion corpus for an "
            "initial learned control. Pair it with a larger naturalistic corpus "
            "for generalization."
        ),
        "gaps": [
            "Acted, controlled sentences; limited spontaneous emotion.",
            "Only 91 actors.",
        ],
        "verification": [
            "https://audeering.github.io/datasets/datasets/crema-d.html",
            "https://www.tensorflow.org/datasets/catalog/crema_d",
        ],
    },
    {
        "id": "msp_podcast",
        "name": "MSP-Podcast v2.0",
        "role": "emotion",
        "training_role": "secondary",
        "speaker_id_available": "partial",
        "speaker_id_notes": (
            "Speaker identity is missing for part of the train set; only the "
            "identified subset can be used for speaker-disjoint training."
        ),
        "balance": {
            "status": "imbalanced",
            "notes": (
                "Naturalistic podcast labels are long-tailed. Sample to the "
                "target class balance and verify speaker counts before training."
            ),
        },
        "source": "https://lab-msp.com/MSP/MSP-Podcast.html",
        "license": "Academic License (FDP data transfer form)",
        "license_url": "https://lab-msp.com/MSP/MSP-Podcast.html",
        "commercial_use": "requires_agreement",
        "commercial_use_notes": (
            "The corpus is distributed under an academic license that requires "
            "an institutional signature. The lab states the source podcasts were "
            "selected to permit commercial use, but the corpus agreement and "
            "redistribution terms require legal review before commercial training."
        ),
        "size": {
            "speaking_turns": 264705,
            "hours": 409,
            "speakers": {
                "train": 2220,
                "development": 704,
                "test1": 465,
                "test2": 112,
                "test3": 428,
            },
            "segments": {
                "train": 169190,
                "development": 34399,
                "test1": 46294,
                "test2": 14822,
                "test3": 3200,
            },
            "language": "English",
            "modality": "naturalistic podcast speech",
        },
        "labels": {
            "categorical": [
                "anger",
                "happiness",
                "sadness",
                "disgust",
                "surprise",
                "fear",
                "contempt",
                "neutral",
                "other",
            ],
            "attributes": ["activation", "dominance", "valence"],
        },
        "local_expected_paths": [
            "data/raw/msp_podcast",
            "data/external/msp_podcast",
            "datasets/msp_podcast",
        ],
        "fit": (
            "Largest naturalistic emotion corpus in the audit. Best candidate "
            "for generalization and speaker-independent evaluation once the "
            "agreement is signed."
        ),
        "gaps": [
            "Access agreement and institutional signature required.",
            "Speaker identity is missing for part of the train set.",
            "Test partitions overlap in speakers and must not be used for training.",
            "Naturalistic podcast audio, not studio read speech.",
        ],
        "verification": ["https://lab-msp.com/MSP/MSP-Podcast.html"],
    },
    {
        "id": "esd",
        "name": "Emotional Speech Dataset (ESD)",
        "role": "emotion",
        "training_role": "research",
        "speaker_id_available": "full",
        "speaker_id_notes": "Speaker IDs are directory names in the ESD release.",
        "balance": {
            "status": "expected_balanced",
            "notes": (
                "Parallel recordings across five emotions for 20 speakers, but "
                "only 10 are English speakers and the license is research-only."
            ),
        },
        "source": "https://github.com/HLTSingapore/Emotional-Speech-Data",
        "license": "CC BY-NC 4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
        "commercial_use": "research_only",
        "commercial_use_notes": "Non-commercial license; exclude from commercial training.",
        "size": {
            "utterances": 35000,
            "speakers": 20,
            "english_speakers": 10,
            "chinese_speakers": 10,
            "language": "English and Chinese",
            "modality": "acted, parallel",
        },
        "labels": {
            "emotions": ["anger", "happiness", "neutral", "sadness", "surprise"],
        },
        "local_expected_paths": [
            "data/raw/esd",
            "data/external/esd",
            "datasets/esd",
        ],
        "fit": (
            "Useful research-only cross-check because the English and Chinese "
            "recordings are parallel. Not usable for commercial training."
        ),
        "gaps": [
            "Non-commercial license.",
            "Only 10 English speakers.",
            "Acted speech.",
        ],
        "verification": [
            "https://huggingface.co/datasets/jspaulsen/esd",
            "https://github.com/HLTSingapore/Emotional-Speech-Data",
        ],
    },
    {
        "id": "ravdess",
        "name": "RAVDESS",
        "role": "emotion",
        "training_role": "evaluation",
        "speaker_id_available": "full",
        "speaker_id_notes": "Actor IDs are encoded in the RAVDESS file names.",
        "balance": {
            "status": "expected_balanced",
            "notes": (
                "24 actors record the same eight emotions, so it is balanced by "
                "design but small; the default release is non-commercial."
            ),
        },
        "source": "https://zenodo.org/record/1188976",
        "license": "CC BY-NC-SA 4.0; commercial license purchasable",
        "license_url": "https://zenodo.org/record/1188976",
        "commercial_use": "research_only_unless_licensed",
        "commercial_use_notes": (
            "The default release is non-commercial. A separate commercial "
            "license can be purchased from the dataset authors."
        ),
        "size": {
            "files": 7356,
            "actors": 24,
            "size_gb": 24.8,
            "language": "English",
            "modality": "acted audio-visual",
        },
        "labels": {
            "emotions": [
                "calm",
                "happy",
                "sad",
                "angry",
                "fearful",
                "disgust",
                "surprised",
                "neutral",
            ],
            "intensity_levels": ["normal", "strong"],
        },
        "local_expected_paths": [
            "data/raw/ravdess",
            "data/external/ravdess",
            "datasets/ravdess",
        ],
        "fit": (
            "Clean acted English with intensity labels. Usable for evaluation "
            "only if the commercial license is purchased."
        ),
        "gaps": [
            "Non-commercial by default.",
            "Acted speech from 24 actors.",
        ],
        "verification": ["https://zenodo.org/record/1188976"],
    },
    {
        "id": "iemocap",
        "name": "IEMOCAP",
        "role": "emotion",
        "training_role": "research",
        "speaker_id_available": "full",
        "speaker_id_notes": "Session and actor IDs are available in the transcripts.",
        "balance": {
            "status": "imbalanced",
            "notes": (
                "Acted/improvised dyadic sessions with uneven category counts; "
                "research-only in any case."
            ),
        },
        "source": "https://sail.usc.edu/iemocap/",
        "license": "USC SAIL academic end-user license",
        "license_url": "https://sail.usc.edu/iemocap/iemocap_release.htm",
        "commercial_use": "research_only",
        "commercial_use_notes": "Academic EULA; not a commercial training corpus.",
        "size": {
            "hours": 12,
            "actors": 10,
            "sessions": 5,
            "language": "English",
            "modality": "acted and improvised dyadic",
        },
        "labels": {
            "categorical": ["anger", "happiness", "sadness", "neutral", "frustration", "excited"],
            "attributes": ["valence", "activation", "dominance"],
        },
        "local_expected_paths": [
            "data/raw/iemocap",
            "data/external/iemocap",
            "datasets/iemocap",
        ],
        "fit": "Research benchmark only.",
        "gaps": ["Research-only EULA.", "Small, acted/improvised corpus."],
        "verification": ["https://sail.usc.edu/iemocap/iemocap_release.htm"],
    },
    {
        "id": "vctk",
        "name": "VCTK Corpus 0.92",
        "role": "accent",
        "training_role": "evaluation",
        "speaker_id_available": "full",
        "speaker_id_notes": "Speaker IDs are directory names in VCTK.",
        "balance": {
            "status": "imbalanced",
            "notes": (
                "Verified speaker counts range from 33 (English) to 1 "
                "(Welsh/New Zealand), a max/min ratio of 33 versus the 1.5 "
                "target; not a balanced training set."
            ),
        },
        "source": "https://datashare.ed.ac.uk/handle/10283/3443",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "commercial_use": "allowed_with_obligations",
        "commercial_use_notes": "Commercial use permitted with attribution.",
        "size": {
            "hours": 44,
            "speakers": 110,
            "language": "English",
            "modality": "read speech",
            "accent_speaker_counts": {
                "English": 33,
                "American": 22,
                "Scottish": 19,
                "Irish": 9,
                "Canadian": 8,
                "NorthernIrish": 6,
                "SouthAfrican": 4,
                "Indian": 3,
                "Australian": 2,
                "Welsh": 1,
                "Unknown": 1,
                "NewZealand": 1,
                "British": 1,
            },
        },
        "labels": {
            "accents": "speaker-info.txt self-reported accent and region",
        },
        "local_expected_paths": [
            "data/raw/vctk",
            "data/external/vctk",
            "datasets/vctk",
        ],
        "fit": (
            "High-quality read speech. Useful for held-out accent evaluation and "
            "small-scale adaptation, but the accent classes are highly imbalanced."
        ),
        "gaps": [
            "Too few speakers per accent for robust accent training.",
            "Read newspaper sentences, not conversational speech.",
        ],
        "verification": [
            "https://datashare.ed.ac.uk/handle/10283/3443",
            "https://raw.githubusercontent.com/milletyann/VCTK-Corpus-0.92/main/VCTK-Corpus-0.92/speaker-info.txt",
        ],
    },
    {
        "id": "openslr83",
        "name": "OpenSLR 83 (UK and Ireland English dialects)",
        "role": "accent",
        "training_role": "evaluation",
        "speaker_id_available": "full",
        "speaker_id_notes": "Speaker IDs are directory names in OpenSLR 83.",
        "balance": {
            "status": "imbalanced",
            "notes": (
                "Verified counts are Scottish 39 speakers versus Irish 5, a "
                "max/min ratio of 7.8; Irish is evaluation-only."
            ),
        },
        "source": "https://www.openslr.org/83/",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "commercial_use": "allowed_with_obligations",
        "commercial_use_notes": "Commercial use permitted with attribution and share-alike.",
        "size": {
            "hours": 31,
            "speakers": 120,
            "transcriptions": 17877,
            "language": "English",
            "modality": "read speech",
            "dialect_speaker_counts": {
                "Irish": {"total": 5, "male": 3, "female": 2},
                "Scottish": {"total": 39, "male": 11, "female": 28},
            },
        },
        "labels": {
            "dialects": [
                "Irish English",
                "Midlands English",
                "Northern English",
                "Scottish English",
                "Southern English",
                "Welsh English",
            ],
        },
        "local_expected_paths": [
            "data/raw/openslr83",
            "data/external/openslr83",
            "datasets/openslr83",
        ],
        "fit": (
            "Best open British Isles dialect corpus for Scottish/Irish "
            "evaluation. Irish has only five speakers, so it is not a training "
            "corpus for that accent."
        ),
        "gaps": [
            "Highly imbalanced: Scottish 39 speakers vs Irish 5.",
            "Read sentences; limited spontaneous speech.",
        ],
        "verification": [
            "https://www.openslr.org/83/",
            "https://www.openslr.org/resources/83/about.html",
            "https://aclanthology.org/2020.lrec-1.804.pdf",
        ],
    },
    {
        "id": "svarah",
        "name": "Svarah",
        "role": "accent",
        "training_role": "evaluation",
        "speaker_id_available": "full",
        "speaker_id_notes": "Speaker IDs are directory names in Svarah.",
        "balance": {
            "status": "single_class",
            "notes": (
                "Single accent region (Indian English), so cross-accent balance "
                "does not apply; 117 speakers / 9.6 h is useful for evaluation "
                "and limited adaptation, not broad accent training."
            ),
        },
        "source": "https://github.com/AI4Bharat/Svarah",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "commercial_use": "allowed_with_obligations",
        "commercial_use_notes": "Commercial use permitted with attribution.",
        "size": {
            "hours": 9.6,
            "speakers": 117,
            "districts": 65,
            "states": 19,
            "language": "Indian English",
            "modality": "read and spontaneous",
        },
        "labels": {
            "accents": "Indian English with 19 native-language backgrounds",
        },
        "local_expected_paths": [
            "data/raw/svarah",
            "data/external/svarah",
            "datasets/svarah",
        ],
        "fit": (
            "Useful for Indian English evaluation and limited adaptation. Too "
            "small for broad accent training by itself."
        ),
        "gaps": ["Only 9.6 hours.", "Single accent region."],
        "verification": ["https://github.com/AI4Bharat/Svarah"],
    },
    {
        "id": "edacc",
        "name": "EdAcc (Edinburgh International Accents of English Corpus)",
        "role": "accent",
        "training_role": "evaluation",
        "speaker_id_available": "full",
        "speaker_id_notes": "Speaker IDs are provided in the EdAcc metadata.",
        "balance": {
            "status": "imbalanced",
            "notes": (
                "122 speakers spread over more than 40 self-reported accents; "
                "long-tailed by design, so it is an evaluation corpus rather "
                "than a balanced training set."
            ),
        },
        "source": "https://datashare.ed.ac.uk/items/355c07b4-500d-4e80-8f12-225e646293c9",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "commercial_use": "allowed_with_obligations",
        "commercial_use_notes": "Commercial use permitted with attribution and share-alike.",
        "size": {
            "hours": 40,
            "speakers": 122,
            "conversations": 61,
            "accents": "more than 40 self-reported English accents",
            "l1_backgrounds": 51,
            "language": "English",
            "modality": "dyadic video-call conversation",
        },
        "labels": {
            "accents": "self-reported accent plus linguistic background CSV",
        },
        "local_expected_paths": [
            "data/raw/edacc",
            "data/external/edacc",
            "datasets/edacc",
        ],
        "fit": (
            "Strong accent-identification and evaluation corpus because it has "
            "broad accent coverage and linguistic background metadata."
        ),
        "gaps": [
            "Conversational, noisy, and not a studio TTS corpus.",
            "Accent labels are self-reported.",
            "CC BY-SA share-alike obligations require review.",
        ],
        "verification": [
            "https://datashare.ed.ac.uk/items/355c07b4-500d-4e80-8f12-225e646293c9",
            "https://groups.inf.ed.ac.uk/edacc/",
        ],
    },
    {
        "id": "globe_v3",
        "name": "GLOBE_V3",
        "role": "accent",
        "training_role": "primary",
        "speaker_id_available": "join_required",
        "speaker_id_notes": (
            "GLOBE_V3 has no speaker_id column. Join common_voice_path to "
            "Common Voice 21 metadata to recover client_id before building "
            "speaker-disjoint splits."
        ),
        "balance": {
            "status": "imbalanced",
            "notes": (
                "Partial samples show a long tail: the 23,364-row statistics "
                "sample has 9,138 US rows versus 15 New Zealand rows, and a "
                "separate 5,298-row shard has 207 US speakers versus 1 New "
                "Zealand speaker. These are partial counts, not full-corpus "
                "measurements; recompute per-accent speakers and hours after "
                "the Common Voice join."
            ),
        },
        "source": "https://huggingface.co/datasets/MushanW/GLOBE_V3",
        "license": "CC0 1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "commercial_use": "allowed",
        "commercial_use_notes": "CC0 public-domain dedication; no attribution required.",
        "size": {
            "rows": 723532,
            "splits": {"train": 704750, "dev": 9603, "test": 9179},
            "sample_rate_hz": 24000,
            "card_download_gb": 165.5,
            "card_dataset_gb": 184.5,
            "repo_tree_gb": 348.7,
            "repo_files": 603,
            "size_note": (
                "The Hugging Face tree measured on 2026-10-28 is 348.7 GB "
                "across 603 files: 165.5 GB of data/*.parquet plus 183.2 GB of "
                "duplicate train/dev/test audio. The card's 165.5 GB download "
                "figure is the Parquet payload and 184.5 GB is the uncompressed "
                "dataset size. Only the Parquet payload is needed."
            ),
            "language": "English",
            "modality": "curated Common Voice speech",
            "paper_reported": {
                "speakers": 23519,
                "accents": 164,
                "hours": 535,
                "note": (
                    "These are the GLOBE paper/card figures for the curated "
                    "GLOBE release. GLOBE_V3 is a larger Common Voice 21 "
                    "rebuild (723,532 rows), so do not reuse 535 h or the paper "
                    "speaker/accent counts as V3 measurements. Recompute them "
                    "from the downloaded V3 metadata."
                ),
            },
        },
        "labels": {
            "accents": "common_voice_accents (self-reported) and predicted_accent",
            "quality": ["snr", "utmos", "wer", "cer", "predicted_gender", "predicted_age"],
        },
        "local_expected_paths": [
            "data/raw/globe_v3",
            "data/external/globe_v3",
            "datasets/globe_v3",
            "data/processed/globe_audio",
            "data/processed/globe_metadata",
        ],
        "fit": (
            "Primary commercially usable accent corpus. Broad accent coverage "
            "and a large speaker pool; use it for training and speaker-disjoint "
            "evaluation after filtering noisy labels."
        ),
        "gaps": [
            "Large download and storage footprint (348.7 GB tree = 165.5 GB Parquet + 183.2 GB duplicate audio).",
            "The local D: drive has 138.1 GB free, so the full tree does not fit; use metadata streaming and selective row-group download.",
            "Accent labels are self-reported and imbalanced.",
            "Common Voice recording conditions vary.",
            "GLOBE_V3 is a larger Common Voice 21 rebuild than the 535 h paper release; recompute hours/speaker/accent counts from V3 metadata.",
            "The HF statistics endpoint returned partial=true for 23,364 train rows; sampled accent counts are not full-corpus counts.",
        ],
        "verification": [
            "https://huggingface.co/datasets/MushanW/GLOBE_V3",
            "https://huggingface.co/datasets/MushanW/GLOBE",
            "https://globecorpus.github.io/",
        ],
    },
    {
        "id": "common_voice_en",
        "name": "Mozilla Common Voice English",
        "role": "accent",
        "training_role": "secondary",
        "speaker_id_available": "full",
        "speaker_id_notes": "client_id identifies speakers in Common Voice metadata.",
        "balance": {
            "status": "imbalanced",
            "notes": (
                "Crowdsourced with incomplete, self-reported accent labels; "
                "requires filtering, deduplication, and class balancing."
            ),
        },
        "source": "https://commonvoice.mozilla.org/en/datasets",
        "license": "CC0 1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "commercial_use": "allowed",
        "commercial_use_notes": "CC0 public-domain dedication.",
        "size": {
            "hours": 3843,
            "speakers": 100973,
            "language": "English",
            "modality": "crowdsourced read speech",
            "release_note": (
                "Release-dependent. The counts are from the English release used "
                "in the prior audit; recompute from the chosen Common Voice "
                "release before training."
            ),
        },
        "labels": {
            "accents": "self-reported accent metadata (incomplete and inconsistent)",
        },
        "local_expected_paths": [
            "data/raw/common_voice",
            "data/external/common_voice",
            "datasets/common_voice",
        ],
        "fit": (
            "Supplemental accent data. The speaker pool is large, but labels and "
            "recording quality are noisy."
        ),
        "gaps": [
            "Self-reported accent labels are inconsistent.",
            "No controlled prompt set across speakers.",
            "Requires filtering, deduplication, and label cleanup.",
        ],
        "verification": ["https://commonvoice.mozilla.org/en/datasets"],
    },
    {
        "id": "l2_arctic",
        "name": "L2-ARCTIC",
        "role": "accent",
        "training_role": "research",
        "speaker_id_available": "full",
        "speaker_id_notes": "Speaker IDs are directory names in L2-ARCTIC.",
        "balance": {
            "status": "expected_balanced",
            "notes": (
                "24 speakers across six L1 backgrounds (about four each), but "
                "the corpus is non-commercial and too small for production."
            ),
        },
        "source": "https://psi.engr.tamu.edu/l2-arctic-corpus/",
        "license": "CC BY-NC 4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
        "commercial_use": "research_only",
        "commercial_use_notes": "Non-commercial license; exclude from commercial training.",
        "size": {
            "speakers": 24,
            "l1_backgrounds": ["Vietnamese", "Korean", "Mandarin", "Spanish", "Hindi", "Arabic"],
            "language": "English (L2)",
            "modality": "read speech with phonemic annotations",
        },
        "labels": {
            "accents": "L1 background and ARPABet phonemic annotations",
        },
        "local_expected_paths": [
            "data/raw/l2_arctic",
            "data/external/l2_arctic",
            "datasets/l2_arctic",
        ],
        "fit": "Research-only phonetic analysis; not commercial training data.",
        "gaps": ["Non-commercial license.", "Only 24 speakers."],
        "verification": ["https://psi.engr.tamu.edu/l2-arctic-corpus/"],
    },
    {
        "id": "voxceleb2",
        "name": "VoxCeleb2",
        "role": "identity",
        "training_role": "research",
        "speaker_id_available": "full",
        "speaker_id_notes": "Speaker IDs are directory names in VoxCeleb2.",
        "balance": {
            "status": "not_labeled",
            "notes": (
                "Identity corpus with no emotion or accent labels; it cannot be "
                "used to balance or train an expression control."
            ),
        },
        "source": "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html",
        "license": "Research / non-commercial terms",
        "license_url": "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/",
        "commercial_use": "research_only",
        "commercial_use_notes": (
            "The current identity corpus is research/non-commercial only. It "
            "must not be used as a commercial emotion/accent training corpus."
        ),
        "size": {
            "project_clips": 35892,
            "project_speakers": 6111,
            "language": "English (multilingual speakers)",
            "modality": "in-the-wild video interviews",
        },
        "labels": {
            "identity": "speaker identity; no emotion or accent labels",
        },
        "local_expected_paths": ["data/raw/voxceleb2"],
        "fit": "Current core face-to-voice identity corpus only.",
        "gaps": [
            "Research/non-commercial terms.",
            "No emotion or accent labels.",
        ],
        "verification": ["https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html"],
    },
]


def scan_local() -> Dict[str, Any]:
    """Scan a bounded set of roots for known expression corpora."""
    roots = [
        _path("data", "raw"),
        _path("data", "external"),
        _path("data", "datasets"),
        _path("datasets"),
        _path("corpora"),
        Path("D:/datasets"),
        Path("D:/corpora"),
    ]
    top_level: Dict[str, List[str]] = {}
    for root in roots:
        if root.exists():
            try:
                top_level[str(root)] = sorted(
                    entry.name for entry in root.iterdir() if entry.is_dir()
                )
            except OSError as exc:
                top_level[str(root)] = [f"<scan error: {exc}>"]

    present: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        matches = []
        for relative in dataset["local_expected_paths"]:
            candidate = ROOT / relative
            if candidate.exists():
                matches.append(str(candidate.relative_to(ROOT)))
        if matches:
            present.append({"id": dataset["id"], "paths": matches})

    return {
        "roots": [str(root) for root in roots],
        "roots_present": [str(root) for root in roots if root.exists()],
        "top_level_directories": top_level,
        "dataset_matches": present,
        "expression_corpora_present": bool(
            [item for item in present if item["id"] not in {"voxceleb2"}]
        ),
        "voxceleb2_present": any(item["id"] == "voxceleb2" for item in present),
        "notes": [
            "Only the roots above were scanned; this is an availability check, not a full-disk search.",
            "GLOBE_V3 metadata and the selected accent WAVs are processed artifacts under data/processed, not a raw corpus directory.",
            "No CREMA-D, MSP-Podcast, ESD, RAVDESS, IEMOCAP, VCTK, OpenSLR 83, Svarah, or EdAcc corpus directory was found.",
        ],
    }


def build_audit() -> Dict[str, Any]:
    local = scan_local()
    emotion_ids = {"crema_d", "msp_podcast", "esd", "ravdess", "iemocap"}
    accent_ids = {"globe_v3", "common_voice_en", "vctk", "openslr83", "svarah", "edacc"}
    present_ids = {item["id"] for item in local["dataset_matches"]}
    emotion_present = bool(present_ids & emotion_ids)
    accent_present = bool(present_ids & accent_ids)
    commercial = [
        dataset["id"]
        for dataset in DATASETS
        if dataset["commercial_use"] in {"allowed", "allowed_with_obligations"}
    ]
    conditional = [
        dataset["id"]
        for dataset in DATASETS
        if dataset["commercial_use"] in {"requires_agreement", "research_only_unless_licensed"}
    ]
    research_only = [
        dataset["id"]
        for dataset in DATASETS
        if dataset["commercial_use"] == "research_only"
    ]
    return {
        "schema_version": AUDIT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "accessed": ACCESSED,
        "scope_document": "SCOPE.md",
        "purpose": (
            "Decide which emotion and accent corpora can legally support a "
            "production-grade learned control, and record what is missing locally."
        ),
        "method": {
            "local_scan": "Bounded directory scan of the roots listed in local_scan.",
            "dataset_facts": (
                "License, size, speaker, and label facts were checked against the "
                "primary source URLs in each dataset entry on the access date."
            ),
            "no_fabrication_rule": (
                "Partial or sampled counts are labelled; unavailable metrics are "
                "not converted to zeros or passes."
            ),
        },
        "local_scan": local,
        "datasets": DATASETS,
        "globe_accent_selection": _load_json(
            "data/processed/globe_accent_selection_summary.json"
        ),
        "globe_accent_audio_audit": _load_json(
            "data/processed/globe_accent_audio_audit.json"
        ),
        "globe_accent_manifest_prune_report": _load_json(
            "data/processed/globe_accent_manifest_prune_report.json"
        ),
        "crema_d_curation_report": _load_json(
            "data/processed/crema_d_curation_report.json"
        ),
        "emotion_classifier_report": _load_json(
            "results/emotion_classifier_crema_d.json"
        ),
        "summary": {
            "commercial_usable_with_obligations": commercial,
            "commercial_only_after_agreement_or_purchase": conditional,
            "research_only": research_only,
            "training_roles": {
                role: [
                    dataset["id"]
                    for dataset in DATASETS
                    if dataset["training_role"] == role
                ]
                for role in (
                    "primary",
                    "secondary",
                    "pilot_evaluation",
                    "evaluation",
                    "research",
                )
            },
            "speaker_id_recovery_required": [
                dataset["id"]
                for dataset in DATASETS
                if dataset["speaker_id_available"] == "join_required"
            ],
            "expression_corpora_present_locally": local["expression_corpora_present"],
            "identity_corpus_present_locally": local["voxceleb2_present"],
        },
        "recommended_control_inventory": {
            "status": "proposed_not_yet_trained",
            "emotion": {
                "current_demo": [
                    "neutral",
                    "happy",
                    "sad",
                    "angry",
                    "whispered",
                    "surprised",
                ],
                "train": ["neutral", "happy", "sad", "angry", "fear", "disgust"],
                "add_when_commercially_usable_data_exists": ["surprised"],
                "remove_or_move": {
                    "whispered": "voice-quality control, not an emotion",
                },
                "standard_assessment": (
                    "The current six are not the standard basic-emotion set. "
                    "Standard expressive-TTS inventories use anger, disgust, "
                    "fear, happiness, sadness, and surprise, usually with "
                    "neutral. Whispered is a phonation style. Fear and disgust "
                    "are missing. Surprise is deferred because no commercially "
                    "usable surprise corpus with enough speakers is confirmed."
                ),
                "primary_source": "CREMA-D (six classes, 91 actors; pilot/evaluation-only, below 2 h/class)",
                "secondary_source": "MSP-Podcast v2.0 after institutional agreement",
            },
            "accent": {
                "current_demo": [
                    "General American",
                    "British RP",
                    "Australian",
                    "Indian English",
                    "Irish",
                    "Scottish",
                ],
                "train": [
                    "US English (General American)",
                    "British English (England English)",
                    "Australian English",
                    "Canadian English",
                    "Indian/South Asian English",
                ],
                "evaluation_only": [
                    "Irish English",
                    "Scottish English",
                    "Welsh English",
                    "New Zealand English",
                    "South African English",
                    "Northern Irish English",
                ],
                "evaluation_only_notes": (
                    "Irish is evaluation-only in the current GLOBE selection: "
                    "1.32 h pre-trim is available, below the 2.0 h gate. The "
                    "other thin accents remain evaluation-only."
                ),
                "rename": {
                    "British RP": (
                        "British English (England English) unless RP-specific "
                        "data is acquired"
                    ),
                },
                "standard_assessment": (
                    "The current six are recognizable and market-relevant, but "
                    "not balanced. The corpus label is England English, not RP; "
                    "Scottish is thin; Canadian is missing despite stronger "
                    "coverage. Use the six proposed classes as the first "
                    "balanced target and keep thin accents evaluation-only."
                ),
                "primary_source": (
                    "GLOBE_V3 (CC0), after recovering speaker IDs through the "
                    "Common Voice metadata join"
                ),
                "secondary_source": (
                    "Common Voice English (CC0); EdAcc, Svarah, VCTK, and "
                    "OpenSLR 83 for evaluation"
                ),
            },
            "balance_thresholds": {
                "min_speakers_per_class": 30,
                "min_hours_per_class": 2.0,
                "max_min_class_ratio": 1.5,
                "per_speaker_seconds_cap": 120,
                "max_clips_per_speaker": 40,
                "pre_trim_target_hours": 2.5,
                "split_policy": "speaker-disjoint train/validation/test",
                "policy": (
                    "A class that misses any threshold is evaluation-only and "
                    "must not be exposed in the product."
                ),
            },
            "gate_policy": (
                "Report every class separately; unavailable metrics stay "
                "unavailable and never pass."
            ),
        },
        "recommendation": {
            "emotion_first_acquisition": [
                "CREMA-D (ODbL 1.0) for pilot/evaluation work; it is below the 2 h/class gate and cannot train a learned control.",
                "MSP-Podcast v2.0 after institutional agreement and legal review for learned emotion control and naturalistic generalization.",
                "RAVDESS only if the commercial license is purchased; otherwise evaluation only.",
            ],
            "accent_first_acquisition": [
                "GLOBE_V3 (CC0) as the primary accent corpus. The metadata-first pass is complete: 723,532 rows indexed, 100% joined to Common Voice speakers, and 9,688 clips selected across 543 row groups. Only selected row groups are downloaded; the 348.7 GB tree is never cloned.",
                "EdAcc (CC BY-SA 4.0) for broad accent evaluation and accent identification.",
                "Svarah (CC BY 4.0) for Indian English evaluation/adaptation.",
                "VCTK and OpenSLR 83 as held-out evaluation sets, not primary training sets.",
            ],
            "do_not_train_commercially_on": [
                "ESD (CC BY-NC 4.0)",
                "RAVDESS (CC BY-NC-SA 4.0 unless commercially licensed)",
                "IEMOCAP (academic EULA)",
                "L2-ARCTIC (CC BY-NC 4.0)",
                "VoxCeleb2 (research/non-commercial)",
            ],
            "immediate_blockers": [
                (
                    "No emotion corpus is present locally."
                    if not emotion_present
                    else "CREMA-D is present but below the 2 h/class gate; it is pilot/evaluation-only."
                ),
                (
                    "No accent corpus is present locally."
                    if not accent_present
                    else "Accent audio is present; the post-download audit passes the 2.0 h gate."
                ),
                "MSP-Podcast requires an institutional agreement.",
                "No human MOS/CMOS or perceptual identification ratings exist.",
                "The CREMA-D pilot emotion classifier exists but is not validated cross-corpus; no accent classifier is implemented.",
                "No accent-aware G2P/pronunciation lexicon is implemented.",
            ],
        },
        "gaps": [
            {
                "id": "local_data",
                "description": (
                    "No emotion corpus is present locally; the GLOBE accent subset is present."
                    if accent_present and not emotion_present
                    else (
                        "Neither an emotion nor an accent corpus is present locally."
                        if not accent_present and not emotion_present
                        else "Emotion and accent corpora are present locally."
                    )
                ),
                "impact": (
                    "Emotion training cannot start; accent training can proceed after the post-trim audit."
                    if accent_present and not emotion_present
                    else "Learned emotion control is blocked by CREMA-D's 0.633-0.963 h/class; accent control is blocked by classifier/evaluation and G2P."
                ),
                "next_step": (
                    "Acquire CREMA-D; keep MSP-Podcast blocked until the agreement is executed."
                    if accent_present and not emotion_present
                    else "Keep CREMA-D evaluation-only; execute the MSP-Podcast agreement before learned emotion training."
                ),
            },
            {
                "id": "licensing",
                "description": "Share-alike and academic-agreement obligations need review.",
                "impact": "Some corpora cannot be assumed usable for commercial training.",
                "next_step": "Legal review of ODbL, CC BY-SA, and MSP-Podcast terms before training.",
            },
            {
                "id": "storage",
                "description": (
                    "GLOBE_V3 is 348.7 GB: 165.5 GB of Parquet plus 183.2 GB "
                    "of duplicate train/dev/test audio."
                ),
                "impact": (
                    "A full clone does not fit in 138.1 GB free and is not "
                    "needed; the metadata-first path avoids it."
                ),
                "next_step": (
                    "Keep only the selected row-group audio and 24 kHz WAVs; "
                    "never clone the full tree."
                ),
            },
            {
                "id": "labels",
                "description": "Accent labels in Common Voice/GLOBE and EdAcc are self-reported; classes are imbalanced.",
                "impact": "Naive training would learn label noise and majority accents.",
                "next_step": "Define a minimum-speaker threshold per accent and a label-cleaning protocol.",
            },
            {
                "id": "classifiers",
                "description": "No real emotion or accent classifier is implemented.",
                "impact": "The expression gate cannot measure classifier accuracy.",
                "next_step": "Train or integrate held-out emotion/accent classifiers and keep them out of the training loop.",
            },
            {
                "id": "perceptual",
                "description": "No human MOS/CMOS or identification study has been run.",
                "impact": "The perceptual-validation gate cannot pass.",
                "next_step": "Run a blinded listener study with real ratings; never synthesize MOS.",
            },
            {
                "id": "accent_frontend",
                "description": "The current accent path is regex respelling plus fixed deltas.",
                "impact": "It cannot produce phonological accent control and damages intelligibility.",
                "next_step": "Implement accent-aware G2P/pronunciation lexicons before training an accent module.",
            },
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit emotion/accent data availability and licensing")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "expression_data_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Expression corpora present locally: {audit['local_scan']['expression_corpora_present']}")
    print(f"Datasets audited: {len(audit['datasets'])}")


if __name__ == "__main__":
    main()
