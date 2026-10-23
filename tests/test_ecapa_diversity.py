"""Test if CFVN produces diverse ECAPA embeddings for different faces after amplification fix."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voix.inference.pipeline import VoixPipeline

pipeline = VoixPipeline(device="cpu", enable_watermarking=False)

test_faces = [
    ("C1_01_young_female_Zendaya.jpg", "Young female"),
    ("C1_02_young_male_Timothee_Chalamet.jpg", "Young male"),
    ("C2_08_south_asian_female_Priyanka_Chopra.jpg", "South Asian female"),
    ("C2_11_black_male_Barack_Obama.jpg", "Black male"),
    ("C2_15_latino_male_Pedro_Pascal.jpg", "Latino male"),
    ("C3_17_prominent_jaw_Arnold_Schwarzenegger.jpg", "Prominent jaw male"),
    ("SD_07_rhotacism_Barbara_Walters.jpg", "Rhoticism female"),
]

text = "The human voice reflects both physical anatomy and lived culture, carrying the unique cadence of personal identity."

ecapa_embeddings = {}
for face_file, desc in test_faces:
    face_path = ROOT / "data" / "test_subjects" / face_file
    if not face_path.exists():
        print(f"SKIP: {face_file} not found")
        continue
    results = pipeline.generate(
        face_input=str(face_path),
        text=text,
        num_voices=1,
        temperature=1.0,
        watermark=False,
    )
    e_s = results[0]["speaker_embedding"].numpy().flatten()
    ecapa_embeddings[desc] = e_s
    print(f"{desc}: ECAPA norm={np.linalg.norm(e_s):.6f}, first 5: {np.round(e_s[:5], 4)}")

print()
print("Pairwise ECAPA cosine similarity:")
names = list(ecapa_embeddings.keys())
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        v1 = ecapa_embeddings[names[i]]
        v2 = ecapa_embeddings[names[j]]
        sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        print(f"  {names[i]:25s} vs {names[j]:25s}: {sim:.6f}")

print("\nDone!")
