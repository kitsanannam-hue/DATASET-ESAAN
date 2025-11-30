#!/usr/bin/env python3
"""
Generate JSONL manifest files for AudioCraft training.
"""

import json
from pathlib import Path
import soundfile as sf


def process_gdrive_audio(raw_dir: Path, processed_dir: Path) -> tuple:
    """Process imported Google Drive audio files."""
    import librosa
    import soundfile as sf
    import numpy as np
    
    gdrive_raw = raw_dir / "gdrive"
    if not gdrive_raw.exists():
        return [], []
    
    musicgen_out = processed_dir / "musicgen" / "gdrive"
    audiogen_out = processed_dir / "audiogen" / "gdrive"
    musicgen_out.mkdir(parents=True, exist_ok=True)
    audiogen_out.mkdir(parents=True, exist_ok=True)
    
    musicgen_data = []
    audiogen_data = []
    
    for audio_file in gdrive_raw.rglob("*.wav"):
        try:
            audio_32k, _ = librosa.load(audio_file, sr=32000, mono=True)
            
            rms = np.sqrt(np.mean(audio_32k**2))
            if rms > 0:
                target_rms = 10 ** (-18.0 / 20)
                audio_32k = audio_32k * (target_rms / rms)
                audio_32k = np.clip(audio_32k, -1.0, 1.0)
            
            max_samples_32k = 30 * 32000
            if len(audio_32k) > max_samples_32k:
                audio_32k = audio_32k[:max_samples_32k]
            
            out_path_32k = musicgen_out / f"{audio_file.stem}.wav"
            sf.write(out_path_32k, audio_32k, 32000)
            
            name_parts = audio_file.stem.replace("_", " ").replace("-", " ")
            if "toei" in audio_file.stem.lower():
                desc = f"Thai toei flute {name_parts}"
            elif "phuthai" in audio_file.stem.lower():
                desc = f"Thai phuthai instrument {name_parts}"
            else:
                desc = f"Thai instrument {name_parts}"
            
            musicgen_data.append({
                "path": str(out_path_32k.absolute()),
                "duration": len(audio_32k) / 32000,
                "sample_rate": 32000,
                "description": desc
            })
            
            audio_16k, _ = librosa.load(audio_file, sr=16000, mono=True)
            
            rms = np.sqrt(np.mean(audio_16k**2))
            if rms > 0:
                target_rms = 10 ** (-18.0 / 20)
                audio_16k = audio_16k * (target_rms / rms)
                audio_16k = np.clip(audio_16k, -1.0, 1.0)
            
            max_samples_16k = 10 * 16000
            if len(audio_16k) > max_samples_16k:
                audio_16k = audio_16k[:max_samples_16k]
            
            out_path_16k = audiogen_out / f"{audio_file.stem}.wav"
            sf.write(out_path_16k, audio_16k, 16000)
            
            audiogen_data.append({
                "path": str(out_path_16k.absolute()),
                "duration": len(audio_16k) / 16000,
                "sample_rate": 16000,
                "description": desc
            })
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
    
    return musicgen_data, audiogen_data


def main():
    """Generate manifest files."""
    processed_dir = Path("data/processed")
    raw_dir = Path("data/raw")
    
    print("=" * 60)
    print("Generating AudioCraft Training Manifests")
    print("=" * 60)
    
    musicgen_data = []
    audiogen_data = []
    
    print("\n--- Processing Google Drive imports ---")
    gdrive_music, gdrive_audio = process_gdrive_audio(raw_dir, processed_dir)
    print(f"Processed {len(gdrive_music)} Google Drive files for MusicGen")
    print(f"Processed {len(gdrive_audio)} Google Drive files for AudioGen")
    musicgen_data.extend(gdrive_music)
    audiogen_data.extend(gdrive_audio)
    
    print("\n--- Collecting MusicGen samples (32kHz) ---")
    musicgen_dirs = [
        processed_dir / "musicgen" / "synthetic",
        processed_dir / "musicgen" / "fma",
        processed_dir / "musicgen" / "attached",
        processed_dir / "musicgen" / "gdrive"
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
        processed_dir / "audiogen" / "attached",
        processed_dir / "audiogen" / "gdrive"
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
