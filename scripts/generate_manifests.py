#!/usr/bin/env python3
"""
Generate JSONL manifest files for AudioCraft training.
"""

import json
from pathlib import Path
import soundfile as sf


def main():
    """Generate manifest files."""
    processed_dir = Path("data/processed")
    
    print("=" * 60)
    print("Generating AudioCraft Training Manifests")
    print("=" * 60)
    
    musicgen_data = []
    audiogen_data = []
    
    print("\n--- Collecting MusicGen samples (32kHz) ---")
    musicgen_dirs = [
        processed_dir / "musicgen" / "synthetic",
        processed_dir / "musicgen" / "fma",
        processed_dir / "musicgen" / "attached"
    ]
    
    for dir_path in musicgen_dirs:
        if dir_path.exists():
            for f in dir_path.glob("*.wav"):
                try:
                    info = sf.info(f)
                    musicgen_data.append({
                        "path": str(f.absolute()),
                        "duration": info.duration,
                        "sample_rate": info.samplerate,
                        "description": f.stem.replace("_", " ").replace("-", " ")
                    })
                except Exception as e:
                    print(f"Error reading {f}: {e}")
    
    print(f"Found {len(musicgen_data)} MusicGen samples")
    
    print("\n--- Collecting AudioGen samples (16kHz) ---")
    audiogen_dirs = [
        processed_dir / "audiogen" / "synthetic",
        processed_dir / "audiogen" / "esc50",
        processed_dir / "audiogen" / "attached"
    ]
    
    esc50_meta = Path("data/raw/esc50/meta/esc50.csv")
    esc50_categories = {}
    if esc50_meta.exists():
        import pandas as pd
        df = pd.read_csv(esc50_meta)
        for _, row in df.iterrows():
            esc50_categories[row['filename']] = row['category']
    
    for dir_path in audiogen_dirs:
        if dir_path.exists():
            for f in dir_path.glob("*.wav"):
                try:
                    info = sf.info(f)
                    
                    if f.name in esc50_categories:
                        category = esc50_categories[f.name]
                        description = f"Sound of {category.replace('_', ' ')}"
                    else:
                        description = f.stem.replace("_", " ").replace("-", " ")
                    
                    audiogen_data.append({
                        "path": str(f.absolute()),
                        "duration": info.duration,
                        "sample_rate": info.samplerate,
                        "description": description
                    })
                except Exception as e:
                    print(f"Error reading {f}: {e}")
    
    print(f"Found {len(audiogen_data)} AudioGen samples")
    
    print("\n--- Writing Manifests ---")
    
    musicgen_manifest = processed_dir / "musicgen_train.jsonl"
    with open(musicgen_manifest, 'w') as f:
        for item in musicgen_data:
            f.write(json.dumps(item) + "\n")
    print(f"MusicGen: {musicgen_manifest} ({len(musicgen_data)} samples)")
    
    audiogen_manifest = processed_dir / "audiogen_train.jsonl"
    with open(audiogen_manifest, 'w') as f:
        for item in audiogen_data:
            f.write(json.dumps(item) + "\n")
    print(f"AudioGen: {audiogen_manifest} ({len(audiogen_data)} samples)")
    
    summary = {
        "musicgen_samples": len(musicgen_data),
        "audiogen_samples": len(audiogen_data),
        "total_samples": len(musicgen_data) + len(audiogen_data),
        "manifests": {
            "musicgen": str(musicgen_manifest),
            "audiogen": str(audiogen_manifest)
        },
        "sample_descriptions": {
            "musicgen": [d["description"] for d in musicgen_data[:5]],
            "audiogen": [d["description"] for d in audiogen_data[:5]]
        }
    }
    
    with open(Path("data/dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Manifest Generation Complete!")
    print("=" * 60)
    print(f"\nMusicGen: {len(musicgen_data)} samples")
    print(f"AudioGen: {len(audiogen_data)} samples")
    print(f"Total: {len(musicgen_data) + len(audiogen_data)} samples")
    
    return summary


if __name__ == "__main__":
    main()
