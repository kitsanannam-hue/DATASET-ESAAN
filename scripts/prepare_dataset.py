#!/usr/bin/env python3
"""
Dataset Preparation Script for AudioCraft Training
Downloads and prepares real audio datasets:
- ESC-50 for AudioGen (environmental sounds)
- FMA-small for MusicGen (music)
- Processes attached audio files
"""

import os
import json
import zipfile
import tarfile
import shutil
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import requests
import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm
import pandas as pd


class DatasetDownloader:
    """Download and extract audio datasets."""
    
    ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    FMA_SMALL_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
    FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.cache_dir = self.data_dir / "cache"
        
        for d in [self.raw_dir, self.processed_dir, self.cache_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, dest_path: Path, desc: str = "Downloading") -> bool:
        """Download a file with progress bar."""
        if dest_path.exists():
            print(f"File already exists: {dest_path}")
            return True
        
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        pbar.update(size)
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            if dest_path.exists():
                dest_path.unlink()
            return False
    
    def download_esc50(self) -> Optional[Path]:
        """Download ESC-50 dataset for AudioGen training."""
        dest_path = self.cache_dir / "esc50.zip"
        extract_path = self.raw_dir / "esc50"
        
        if extract_path.exists():
            print(f"ESC-50 already extracted at: {extract_path}")
            return extract_path
        
        print("Downloading ESC-50 dataset...")
        if not self.download_file(self.ESC50_URL, dest_path, "ESC-50"):
            return None
        
        print("Extracting ESC-50...")
        with zipfile.ZipFile(dest_path, 'r') as zf:
            zf.extractall(self.raw_dir)
        
        extracted = self.raw_dir / "ESC-50-master"
        if extracted.exists():
            extracted.rename(extract_path)
        
        return extract_path
    
    def download_fma_small(self) -> Optional[Path]:
        """Download FMA-small dataset for MusicGen training."""
        dest_path = self.cache_dir / "fma_small.zip"
        meta_path = self.cache_dir / "fma_metadata.zip"
        extract_path = self.raw_dir / "fma_small"
        
        if extract_path.exists() and (extract_path / "000").exists():
            print(f"FMA-small already extracted at: {extract_path}")
            return extract_path
        
        print("Downloading FMA-small dataset (this may take a while, ~8GB)...")
        if not self.download_file(self.FMA_SMALL_URL, dest_path, "FMA-small"):
            print("FMA download failed. Will use alternative smaller dataset.")
            return None
        
        print("Downloading FMA metadata...")
        self.download_file(self.FMA_METADATA_URL, meta_path, "FMA Metadata")
        
        print("Extracting FMA-small...")
        extract_path.mkdir(exist_ok=True)
        with zipfile.ZipFile(dest_path, 'r') as zf:
            zf.extractall(extract_path)
        
        if meta_path.exists():
            with zipfile.ZipFile(meta_path, 'r') as zf:
                zf.extractall(self.raw_dir / "fma_metadata")
        
        return extract_path


class AudioProcessor:
    """Process and convert audio files for AudioCraft training."""
    
    MUSICGEN_SR = 32000
    AUDIOGEN_SR = 16000
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        (self.processed_dir / "musicgen").mkdir(parents=True, exist_ok=True)
        (self.processed_dir / "audiogen").mkdir(parents=True, exist_ok=True)
    
    def process_audio(
        self,
        input_path: Path,
        output_path: Path,
        target_sr: int,
        max_duration: float = 30.0,
        normalize: bool = True,
        target_db: float = -18.0
    ) -> Optional[Dict]:
        """Process a single audio file."""
        try:
            audio, sr = librosa.load(input_path, sr=target_sr, mono=True)
            
            max_samples = int(max_duration * target_sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            
            if normalize:
                rms = np.sqrt(np.mean(audio**2))
                if rms > 0:
                    target_rms = 10 ** (target_db / 20)
                    audio = audio * (target_rms / rms)
                    audio = np.clip(audio, -1.0, 1.0)
            
            sf.write(output_path, audio, target_sr)
            
            duration = len(audio) / target_sr
            return {
                "path": str(output_path),
                "duration": duration,
                "sample_rate": target_sr,
                "original_path": str(input_path)
            }
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return None
    
    def process_esc50(self, esc50_path: Path) -> List[Dict]:
        """Process ESC-50 dataset for AudioGen."""
        audio_dir = esc50_path / "audio"
        meta_file = esc50_path / "meta" / "esc50.csv"
        output_dir = self.processed_dir / "audiogen" / "esc50"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not audio_dir.exists():
            print(f"ESC-50 audio directory not found: {audio_dir}")
            return []
        
        df = pd.read_csv(meta_file) if meta_file.exists() else None
        
        results = []
        audio_files = list(audio_dir.glob("*.wav"))
        
        print(f"Processing {len(audio_files)} ESC-50 audio files...")
        for audio_file in tqdm(audio_files, desc="Processing ESC-50"):
            output_path = output_dir / audio_file.name
            result = self.process_audio(
                audio_file,
                output_path,
                target_sr=self.AUDIOGEN_SR,
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
                        result['category'] = "unknown"
                else:
                    result['description'] = "Environmental sound"
                    result['category'] = "unknown"
                results.append(result)
        
        return results
    
    def process_fma_small(self, fma_path: Path, max_files: int = 500) -> List[Dict]:
        """Process FMA-small dataset for MusicGen."""
        output_dir = self.processed_dir / "musicgen" / "fma"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_path = self.data_dir / "raw" / "fma_metadata" / "fma_metadata"
        tracks_path = metadata_path / "tracks.csv" if metadata_path.exists() else None
        
        genres_df = None
        if tracks_path and tracks_path.exists():
            try:
                genres_df = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
            except:
                genres_df = None
        
        audio_files = []
        for subdir in fma_path.iterdir():
            if subdir.is_dir() and subdir.name.isdigit():
                audio_files.extend(subdir.glob("*.mp3"))
        
        audio_files = list(audio_files)[:max_files]
        
        results = []
        print(f"Processing {len(audio_files)} FMA audio files...")
        for audio_file in tqdm(audio_files, desc="Processing FMA"):
            track_id = int(audio_file.stem)
            output_path = output_dir / f"{track_id}.wav"
            
            result = self.process_audio(
                audio_file,
                output_path,
                target_sr=self.MUSICGEN_SR,
                max_duration=30.0
            )
            
            if result:
                genre = "music"
                if genres_df is not None:
                    try:
                        genre = genres_df.loc[track_id, ('track', 'genre_top')]
                        if pd.isna(genre):
                            genre = "music"
                    except:
                        genre = "music"
                
                result['description'] = f"{genre} music track"
                result['genre'] = genre
                results.append(result)
        
        return results
    
    def process_attached_files(self, attached_dir: Path) -> Tuple[List[Dict], List[Dict]]:
        """Process attached audio files."""
        musicgen_dir = self.processed_dir / "musicgen" / "attached"
        audiogen_dir = self.processed_dir / "audiogen" / "attached"
        musicgen_dir.mkdir(parents=True, exist_ok=True)
        audiogen_dir.mkdir(parents=True, exist_ok=True)
        
        musicgen_results = []
        audiogen_results = []
        
        audio_files = list(attached_dir.glob("*.wav"))
        
        print(f"Processing {len(audio_files)} attached audio files...")
        for i, audio_file in enumerate(tqdm(audio_files, desc="Processing attached")):
            music_output = musicgen_dir / f"attached_{i+1}.wav"
            music_result = self.process_audio(
                audio_file,
                music_output,
                target_sr=self.MUSICGEN_SR,
                max_duration=30.0
            )
            if music_result:
                music_result['description'] = f"Audio sample {i+1}"
                musicgen_results.append(music_result)
            
            audio_output = audiogen_dir / f"attached_{i+1}.wav"
            audio_result = self.process_audio(
                audio_file,
                audio_output,
                target_sr=self.AUDIOGEN_SR,
                max_duration=10.0
            )
            if audio_result:
                audio_result['description'] = f"Audio sample {i+1}"
                audiogen_results.append(audio_result)
        
        return musicgen_results, audiogen_results


class ManifestGenerator:
    """Generate JSONL manifest files for AudioCraft training."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
    
    def generate_manifest(self, data: List[Dict], output_path: Path):
        """Generate a JSONL manifest file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for item in data:
                manifest_item = {
                    "path": item["path"],
                    "duration": item["duration"],
                    "sample_rate": item["sample_rate"],
                    "description": item.get("description", "")
                }
                f.write(json.dumps(manifest_item) + "\n")
        
        print(f"Generated manifest: {output_path} ({len(data)} entries)")
    
    def generate_all_manifests(
        self,
        musicgen_data: List[Dict],
        audiogen_data: List[Dict]
    ):
        """Generate all manifest files."""
        musicgen_path = self.processed_dir / "musicgen_train.jsonl"
        self.generate_manifest(musicgen_data, musicgen_path)
        
        audiogen_path = self.processed_dir / "audiogen_train.jsonl"
        self.generate_manifest(audiogen_data, audiogen_path)
        
        combined = musicgen_data + audiogen_data
        combined_path = self.processed_dir / "combined_train.jsonl"
        self.generate_manifest(combined, combined_path)
        
        return {
            "musicgen": str(musicgen_path),
            "audiogen": str(audiogen_path),
            "combined": str(combined_path)
        }


def main():
    """Main function to prepare datasets."""
    print("=" * 60)
    print("AudioCraft Dataset Preparation")
    print("=" * 60)
    
    downloader = DatasetDownloader("data")
    processor = AudioProcessor("data")
    manifest_gen = ManifestGenerator("data")
    
    all_musicgen_data = []
    all_audiogen_data = []
    
    attached_dir = Path("attached_assets")
    if attached_dir.exists():
        print("\n--- Processing Attached Files ---")
        music_attached, audio_attached = processor.process_attached_files(attached_dir)
        all_musicgen_data.extend(music_attached)
        all_audiogen_data.extend(audio_attached)
        print(f"Processed {len(music_attached)} attached files for MusicGen")
        print(f"Processed {len(audio_attached)} attached files for AudioGen")
    
    print("\n--- Downloading ESC-50 Dataset ---")
    esc50_path = downloader.download_esc50()
    if esc50_path:
        print("\n--- Processing ESC-50 ---")
        esc50_data = processor.process_esc50(esc50_path)
        all_audiogen_data.extend(esc50_data)
        print(f"Processed {len(esc50_data)} ESC-50 files")
    
    print("\n--- Generating Manifest Files ---")
    manifests = manifest_gen.generate_all_manifests(
        all_musicgen_data,
        all_audiogen_data
    )
    
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"\nMusicGen samples: {len(all_musicgen_data)}")
    print(f"AudioGen samples: {len(all_audiogen_data)}")
    print(f"\nManifest files:")
    for name, path in manifests.items():
        print(f"  - {name}: {path}")
    
    summary = {
        "musicgen_samples": len(all_musicgen_data),
        "audiogen_samples": len(all_audiogen_data),
        "manifests": manifests
    }
    
    with open("data/dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: data/dataset_summary.json")
    
    return summary


if __name__ == "__main__":
    main()
