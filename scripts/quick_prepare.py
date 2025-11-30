#!/usr/bin/env python3
"""
Quick Dataset Preparation - Processes a subset of data for faster results.
"""

import os
import json
from pathlib import Path
from typing import List, Dict
import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm
import pandas as pd


def process_audio(
    input_path: Path,
    output_path: Path,
    target_sr: int,
    max_duration: float = 30.0
) -> Dict:
    """Process a single audio file."""
    try:
        audio, sr = librosa.load(input_path, sr=target_sr, mono=True)
        
        max_samples = int(max_duration * target_sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            target_rms = 10 ** (-18.0 / 20)
            audio = audio * (target_rms / rms)
            audio = np.clip(audio, -1.0, 1.0)
        
        sf.write(output_path, audio, target_sr)
        
        return {
            "path": str(output_path.absolute()),
            "duration": len(audio) / target_sr,
            "sample_rate": target_sr
        }
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return None


def main():
    """Quick dataset preparation."""
    print("=" * 60)
    print("Quick AudioCraft Dataset Preparation")
    print("=" * 60)
    
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    raw_dir = data_dir / "raw"
    
    musicgen_dir = processed_dir / "musicgen"
    audiogen_dir = processed_dir / "audiogen"
    musicgen_dir.mkdir(parents=True, exist_ok=True)
    audiogen_dir.mkdir(parents=True, exist_ok=True)
    
    musicgen_data = []
    audiogen_data = []
    
    synth_music = processed_dir / "musicgen" / "synthetic"
    synth_audio = processed_dir / "audiogen" / "synthetic"
    
    if synth_music.exists():
        for f in synth_music.glob("*.wav"):
            try:
                info = sf.info(f)
                musicgen_data.append({
                    "path": str(f.absolute()),
                    "duration": info.duration,
                    "sample_rate": info.samplerate,
                    "description": f.stem.replace("_", " ")
                })
            except:
                pass
    
    if synth_audio.exists():
        for f in synth_audio.glob("*.wav"):
            try:
                info = sf.info(f)
                audiogen_data.append({
                    "path": str(f.absolute()),
                    "duration": info.duration,
                    "sample_rate": info.samplerate,
                    "description": f.stem.replace("_", " ")
                })
            except:
                pass
    
    print(f"Found {len(musicgen_data)} synthetic music samples")
    print(f"Found {len(audiogen_data)} synthetic audio samples")
    
    esc50_audio = raw_dir / "esc50" / "audio"
    esc50_meta = raw_dir / "esc50" / "meta" / "esc50.csv"
    
    if esc50_audio.exists():
        print("\n--- Processing ESC-50 (first 200 samples) ---")
        
        df = None
        if esc50_meta.exists():
            df = pd.read_csv(esc50_meta)
        
        esc50_out = audiogen_dir / "esc50"
        esc50_out.mkdir(parents=True, exist_ok=True)
        
        audio_files = sorted(list(esc50_audio.glob("*.wav")))[:200]
        
        for audio_file in tqdm(audio_files, desc="Processing ESC-50"):
            output_path = esc50_out / audio_file.name
            
            result = process_audio(
                audio_file,
                output_path,
                target_sr=16000,
                max_duration=10.0
            )
            
            if result:
                if df is not None:
                    row = df[df['filename'] == audio_file.name]
                    if not row.empty:
                        category = row.iloc[0]['category']
                        result['description'] = f"Sound of {category.replace('_', ' ')}"
                        result['category'] = category
                    else:
                        result['description'] = "Environmental sound"
                else:
                    result['description'] = "Environmental sound"
                
                audiogen_data.append(result)
        
        print(f"Processed {len(audio_files)} ESC-50 files")
    
    print("\n--- Generating Manifests ---")
    
    musicgen_manifest = processed_dir / "musicgen_train.jsonl"
    with open(musicgen_manifest, 'w') as f:
        for item in musicgen_data:
            f.write(json.dumps(item) + "\n")
    print(f"MusicGen manifest: {musicgen_manifest} ({len(musicgen_data)} samples)")
    
    audiogen_manifest = processed_dir / "audiogen_train.jsonl"
    with open(audiogen_manifest, 'w') as f:
        for item in audiogen_data:
            f.write(json.dumps(item) + "\n")
    print(f"AudioGen manifest: {audiogen_manifest} ({len(audiogen_data)} samples)")
    
    summary = {
        "musicgen_samples": len(musicgen_data),
        "audiogen_samples": len(audiogen_data),
        "total_samples": len(musicgen_data) + len(audiogen_data),
        "manifests": {
            "musicgen": str(musicgen_manifest),
            "audiogen": str(audiogen_manifest)
        }
    }
    
    with open(data_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"\nMusicGen samples: {len(musicgen_data)}")
    print(f"AudioGen samples: {len(audiogen_data)}")
    print(f"Total: {len(musicgen_data) + len(audiogen_data)} samples")
    
    return summary


if __name__ == "__main__":
    main()
