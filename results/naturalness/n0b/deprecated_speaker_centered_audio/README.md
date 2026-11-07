# Deprecated `speaker_centered` N0b audio

These 144 WAVs are from the retired `speaker_centered` condition. They are kept
only for audit, not as a result.

The prototype was built by subtracting each training speaker's mean style and
then averaging the centered rows. Each speaker's centered rows sum to zero, so
the resulting prototype is mathematically zero (measured norm ~1.0e-7). With
`alpha > 0`, the blend therefore only scaled the base prosody half by
`(1 - alpha)`; it did not apply a speaker-centered AMI prosody direction.

The replacement condition is `speaker_balanced` in
`scripts/run_n0b_prosody_prototype.py`: the mean of per-speaker mean
`style[128:]`, with equal weight per training speaker. The current
`results/naturalness/n0b/clips.csv` and `audio/` contain only `mean` and
`speaker_balanced` rows. Do not present these deprecated files as evidence for
a speaker-centered prototype.
