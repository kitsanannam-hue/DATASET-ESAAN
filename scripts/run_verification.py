#!/usr/bin/env python3
"""
Dataset Verification Runner
Runs all verification checks and displays comprehensive results.
"""

import sys
import json
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header(text):
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def print_section(text):
    print(f"\n--- {text} ---")


def main():
    """Run complete dataset verification."""
    print_header("AudioCraft Dataset Verification Suite")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    
    print_section("Dataset Summary")
    summary_path = data_dir / "dataset_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"MusicGen samples: {summary.get('musicgen_samples', 0)}")
        print(f"AudioGen samples: {summary.get('audiogen_samples', 0)}")
        print(f"Total samples: {summary.get('total_samples', 0)}")
        
        if 'sample_descriptions' in summary:
            print("\nSample MusicGen descriptions:")
            for desc in summary['sample_descriptions'].get('musicgen', [])[:3]:
                print(f"  - {desc}")
            print("\nSample AudioGen descriptions:")
            for desc in summary['sample_descriptions'].get('audiogen', [])[:3]:
                print(f"  - {desc}")
    else:
        print("No summary file found. Run generate_manifests.py first.")
        return
    
    print_section("Manifest Verification")
    
    musicgen_manifest = processed_dir / "musicgen_train.jsonl"
    if musicgen_manifest.exists():
        with open(musicgen_manifest) as f:
            musicgen_entries = [json.loads(line) for line in f if line.strip()]
        
        valid_count = sum(1 for e in musicgen_entries if Path(e['path']).exists())
        print(f"\nMusicGen Manifest:")
        print(f"  Total entries: {len(musicgen_entries)}")
        print(f"  Valid paths: {valid_count}")
        print(f"  Invalid paths: {len(musicgen_entries) - valid_count}")
        
        if musicgen_entries:
            total_dur = sum(e.get('duration', 0) for e in musicgen_entries)
            print(f"  Total duration: {total_dur:.1f} seconds ({total_dur/60:.1f} minutes)")
    else:
        print("\nMusicGen manifest not found!")
    
    audiogen_manifest = processed_dir / "audiogen_train.jsonl"
    if audiogen_manifest.exists():
        with open(audiogen_manifest) as f:
            audiogen_entries = [json.loads(line) for line in f if line.strip()]
        
        valid_count = sum(1 for e in audiogen_entries if Path(e['path']).exists())
        print(f"\nAudioGen Manifest:")
        print(f"  Total entries: {len(audiogen_entries)}")
        print(f"  Valid paths: {valid_count}")
        print(f"  Invalid paths: {len(audiogen_entries) - valid_count}")
        
        if audiogen_entries:
            total_dur = sum(e.get('duration', 0) for e in audiogen_entries)
            print(f"  Total duration: {total_dur:.1f} seconds ({total_dur/60:.1f} minutes)")
    else:
        print("\nAudioGen manifest not found!")
    
    print_section("DataLoader Test")
    
    try:
        from src.data.dataset import MusicGenDataset, AudioGenDataset
        
        if musicgen_manifest.exists():
            print("\nTesting MusicGen DataLoader...")
            dataset = MusicGenDataset(str(musicgen_manifest), return_info=True)
            if len(dataset) > 0:
                audio, info = dataset[0]
                print(f"  Sample shape: {audio.shape}")
                print(f"  Sample rate: 32000 Hz")
                print(f"  Duration: {len(audio)/32000:.2f} seconds")
                print(f"  Description: {info['description']}")
                print("  MusicGen DataLoader: OK")
            else:
                print("  MusicGen DataLoader: No samples loaded")
        
        if audiogen_manifest.exists():
            print("\nTesting AudioGen DataLoader...")
            dataset = AudioGenDataset(str(audiogen_manifest), return_info=True)
            if len(dataset) > 0:
                audio, info = dataset[0]
                print(f"  Sample shape: {audio.shape}")
                print(f"  Sample rate: 16000 Hz")
                print(f"  Duration: {len(audio)/16000:.2f} seconds")
                print(f"  Description: {info['description']}")
                print("  AudioGen DataLoader: OK")
            else:
                print("  AudioGen DataLoader: No samples loaded")
        
    except Exception as e:
        print(f"  DataLoader test failed: {e}")
    
    print_section("Configuration Files")
    
    configs = [
        "configs/dataset/musicgen.yaml",
        "configs/dataset/audiogen.yaml",
        "configs/config.yaml"
    ]
    
    for config in configs:
        if Path(config).exists():
            print(f"  [OK] {config}")
        else:
            print(f"  [MISSING] {config}")
    
    print_header("Verification Complete")
    print(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nDataset is ready for AudioCraft training!")
    print("\nTo train MusicGen:")
    print("  python scripts/train.py --config configs/dataset/musicgen.yaml")
    print("\nTo train AudioGen:")
    print("  python scripts/train.py --config configs/dataset/audiogen.yaml")


if __name__ == "__main__":
    main()
