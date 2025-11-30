# AudioCraft Dataset Training Project

## Overview
This project prepares real audio datasets for training AudioCraft models (MusicGen and AudioGen). It downloads, processes, and organizes audio data with proper manifests for training.

## Current State
- **Dataset Ready**: 2023 total samples prepared
  - MusicGen: 13 samples (synthetic music chords and sweeps)
  - AudioGen: 2010 samples (ESC-50 environmental sounds)
- **Manifests Generated**: JSONL format with paths, durations, and descriptions
- **Data Loaders Working**: PyTorch DataLoader verified for both model types

## Project Structure
```
data/
├── processed/
│   ├── musicgen/           # 32kHz music samples
│   │   └── synthetic/      # Generated chord progressions
│   ├── audiogen/           # 16kHz audio samples
│   │   ├── synthetic/      # Generated noise and tones
│   │   └── esc50/         # Environmental sounds
│   ├── musicgen_train.jsonl   # MusicGen training manifest
│   └── audiogen_train.jsonl   # AudioGen training manifest
├── raw/
│   └── esc50/             # ESC-50 dataset (2000 files)
└── dataset_summary.json   # Dataset statistics

scripts/
├── prepare_dataset.py     # Full dataset preparation
├── quick_prepare.py       # Quick dataset prep (subset)
├── generate_manifests.py  # Generate JSONL manifests
├── verify_dataset.py      # Verify dataset integrity
├── run_verification.py    # Comprehensive verification
└── download_free_audio.py # Synthetic audio generation

configs/
├── dataset/
│   ├── musicgen.yaml      # MusicGen dataset config
│   └── audiogen.yaml      # AudioGen dataset config
└── config.yaml            # Main configuration

src/data/
├── dataset.py             # PyTorch Dataset classes
├── loaders.py             # Data loading utilities
└── processors.py          # Audio processing
```

## Key Commands
- **Verify Dataset**: `python scripts/run_verification.py`
- **Generate Manifests**: `python scripts/generate_manifests.py`
- **Quick Prepare**: `python scripts/quick_prepare.py`

## Dataset Details

### MusicGen (32kHz)
- Sample rate: 32000 Hz
- Max duration: 30 seconds
- Content: Synthetic chord progressions and frequency sweeps

### AudioGen (16kHz)
- Sample rate: 16000 Hz
- Max duration: 10 seconds
- Content: ESC-50 environmental sounds (50 categories)
  - Animals, nature sounds, urban sounds, domestic sounds, etc.

## Training Instructions
```bash
# Train MusicGen
python scripts/train.py --config configs/dataset/musicgen.yaml

# Train AudioGen
python scripts/train.py --config configs/dataset/audiogen.yaml
```

## Recent Changes
- 2025-11-30: Initial dataset setup with ESC-50 and synthetic audio
- Created data processing pipeline for audio resampling and normalization
- Generated JSONL manifests with text descriptions for conditioning
- Verified data loaders work correctly with both model types

## User Preferences
- Use real data (not mock data)
- Audio adjusted to correct sample rates (32kHz/16kHz)
- JSONL manifest format with descriptions

## Google Drive Integration
Import audio files from Google Drive or export datasets for backup:

```bash
# List audio files in Google Drive
python scripts/google_drive_sync.py list

# Import audio files from Google Drive
python scripts/google_drive_sync.py import

# Import from specific folder (use folder ID from list command)
python scripts/google_drive_sync.py import --folder-id YOUR_FOLDER_ID

# Export dataset manifests to Google Drive
python scripts/google_drive_sync.py export

# Export with audio files included
python scripts/google_drive_sync.py export --include-audio
```

## Dependencies
- torch, torchaudio
- librosa, soundfile
- numpy, scipy
- pandas (for ESC-50 metadata)
- Google Drive API (via Replit connector)
