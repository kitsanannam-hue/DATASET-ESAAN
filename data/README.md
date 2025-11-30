
# Data Directory

## Overview

This directory contains all datasets used for training, validation, and testing AudioCraft AI agents.

## Structure

- **raw/**: Original, immutable data files (not tracked in git)
- **processed/**: Cleaned, processed data ready for model consumption
- **external/**: External datasets or reference data
- **interim/**: Intermediate processing artifacts
- **cache/**: Cached computations (embeddings, features)

## Dataset Manifest Format

Datasets are described using JSONL (JSON Lines) format, following AudioCraft conventions:

```json
{"path": "/path/to/audio.wav", "duration": 30.5, "sample_rate": 32000, "description": "A funky groove"}
{"path": "/path/to/audio2.wav", "duration": 25.0, "sample_rate": 32000, "description": "Ambient soundscape"}
```

## Adding New Data

1. Place raw files in `data/raw/`
2. Run preprocessing scripts to generate manifests
3. Processed files go to `data/processed/`
4. Update dataset configs in `configs/dataset/`

## Audio Requirements

- **MusicGen**: 32kHz, mono/stereo
- **AudioGen**: 16kHz, mono
- **JASCO**: 32kHz with temporal controls (chords, drums, melody)

Refer to the AudioCraft documentation for specific requirements.
