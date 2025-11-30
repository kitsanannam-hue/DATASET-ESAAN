#!/usr/bin/env python3
"""
Verify Dataset Script
Validates the prepared dataset and displays statistics.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import MusicGenDataset, AudioGenDataset, create_dataloader, verify_dataset


def main():
    """Main verification function."""
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    
    print("=" * 60)
    print("AudioCraft Dataset Verification")
    print("=" * 60)
    
    musicgen_manifest = processed_dir / "musicgen_train.jsonl"
    if musicgen_manifest.exists():
        print("\n--- MusicGen Dataset ---")
        stats = verify_dataset(str(musicgen_manifest), "musicgen")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Valid samples: {stats['valid_samples']}")
        print(f"Invalid samples: {stats['invalid_samples']}")
        print(f"Total duration: {stats['total_duration']:.1f} seconds")
        print(f"Unique descriptions: {stats['unique_descriptions']}")
        
        print("\n--- Testing MusicGen DataLoader ---")
        loader = create_dataloader(
            str(musicgen_manifest),
            dataset_type="musicgen",
            batch_size=2,
            num_workers=0,
            return_info=True
        )
        
        batch = next(iter(loader))
        audio, info = batch
        print(f"Batch shape: {audio.shape}")
        print(f"Sample rate: 32000 Hz")
        print(f"Sample descriptions: {info['description'][:2]}")
    else:
        print("\nMusicGen manifest not found. Run prepare_dataset.py first.")
    
    audiogen_manifest = processed_dir / "audiogen_train.jsonl"
    if audiogen_manifest.exists():
        print("\n--- AudioGen Dataset ---")
        stats = verify_dataset(str(audiogen_manifest), "audiogen")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Valid samples: {stats['valid_samples']}")
        print(f"Invalid samples: {stats['invalid_samples']}")
        print(f"Total duration: {stats['total_duration']:.1f} seconds")
        print(f"Unique descriptions: {stats['unique_descriptions']}")
        
        print("\n--- Testing AudioGen DataLoader ---")
        loader = create_dataloader(
            str(audiogen_manifest),
            dataset_type="audiogen",
            batch_size=2,
            num_workers=0,
            return_info=True
        )
        
        batch = next(iter(loader))
        audio, info = batch
        print(f"Batch shape: {audio.shape}")
        print(f"Sample rate: 16000 Hz")
        print(f"Sample descriptions: {info['description'][:2]}")
    else:
        print("\nAudioGen manifest not found. Run prepare_dataset.py first.")
    
    summary_path = data_dir / "dataset_summary.json"
    if summary_path.exists():
        print("\n--- Dataset Summary ---")
        with open(summary_path) as f:
            summary = json.load(f)
        print(json.dumps(summary, indent=2))
    
    print("\n" + "=" * 60)
    print("Verification Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
