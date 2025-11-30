#!/usr/bin/env python3
"""
Download free audio samples for training AudioCraft models.
Uses Free Music Archive (FMA) sample pack and ESC-50 dataset.
Also generates synthetic audio samples for training diversity.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
import requests
import soundfile as sf
import numpy as np
from tqdm import tqdm


FREESOUND_SAMPLE_URLS = [
    ("rain", "https://cdn.freesound.org/previews/531/531947_1415754-lq.mp3"),
    ("thunder", "https://cdn.freesound.org/previews/316/316840_5766781-lq.mp3"),
    ("birds", "https://cdn.freesound.org/previews/531/531949_1415754-lq.mp3"),
    ("wind", "https://cdn.freesound.org/previews/244/244929_4486188-lq.mp3"),
    ("water", "https://cdn.freesound.org/previews/398/398808_7691025-lq.mp3"),
    ("fire", "https://cdn.freesound.org/previews/346/346116_5121236-lq.mp3"),
    ("footsteps", "https://cdn.freesound.org/previews/336/336598_4939433-lq.mp3"),
    ("door", "https://cdn.freesound.org/previews/411/411642_5121236-lq.mp3"),
    ("typing", "https://cdn.freesound.org/previews/444/444659_7034990-lq.mp3"),
    ("clock", "https://cdn.freesound.org/previews/254/254316_4597519-lq.mp3"),
]

MUSIC_SAMPLE_URLS = [
]


class SyntheticAudioGenerator:
    """Generate synthetic audio samples for training."""
    
    @staticmethod
    def generate_tone(freq: float, duration: float, sr: int = 32000) -> np.ndarray:
        """Generate a pure tone."""
        t = np.linspace(0, duration, int(sr * duration), False)
        return 0.3 * np.sin(2 * np.pi * freq * t)
    
    @staticmethod
    def generate_chord(freqs: List[float], duration: float, sr: int = 32000) -> np.ndarray:
        """Generate a chord from multiple frequencies."""
        t = np.linspace(0, duration, int(sr * duration), False)
        chord = np.zeros_like(t)
        for freq in freqs:
            chord += 0.2 * np.sin(2 * np.pi * freq * t)
        return chord / len(freqs)
    
    @staticmethod
    def generate_noise(duration: float, sr: int = 16000, noise_type: str = "white") -> np.ndarray:
        """Generate noise (white, pink, brown)."""
        samples = int(sr * duration)
        if noise_type == "white":
            return 0.1 * np.random.randn(samples)
        elif noise_type == "pink":
            white = np.random.randn(samples)
            from scipy.signal import lfilter
            b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
            a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
            return 0.1 * lfilter(b, a, white)
        elif noise_type == "brown":
            white = np.random.randn(samples)
            brown = np.cumsum(white)
            brown = brown / np.max(np.abs(brown)) * 0.1
            return brown
        return 0.1 * np.random.randn(samples)
    
    @staticmethod
    def generate_sweep(f_start: float, f_end: float, duration: float, sr: int = 32000) -> np.ndarray:
        """Generate a frequency sweep."""
        t = np.linspace(0, duration, int(sr * duration), False)
        freq = f_start + (f_end - f_start) * t / duration
        phase = 2 * np.pi * np.cumsum(freq) / sr
        return 0.3 * np.sin(phase)
    
    @staticmethod
    def apply_envelope(audio: np.ndarray, attack: float = 0.1, release: float = 0.1) -> np.ndarray:
        """Apply attack-release envelope."""
        sr = len(audio)
        attack_samples = int(attack * sr / 3)
        release_samples = int(release * sr / 3)
        
        envelope = np.ones(len(audio))
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        if release_samples > 0:
            envelope[-release_samples:] = np.linspace(1, 0, release_samples)
        
        return audio * envelope


def generate_synthetic_dataset(output_dir: Path, sr_music: int = 32000, sr_audio: int = 16000) -> tuple:
    """Generate synthetic audio samples for training."""
    gen = SyntheticAudioGenerator()
    
    music_dir = output_dir / "musicgen" / "synthetic"
    audio_dir = output_dir / "audiogen" / "synthetic"
    music_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    music_data = []
    audio_data = []
    
    music_samples = [
        ("c_major_chord", [261.63, 329.63, 392.00], 5.0, "C major chord, warm and bright"),
        ("g_major_chord", [196.00, 246.94, 293.66], 5.0, "G major chord, open and resonant"),
        ("am_chord", [220.00, 261.63, 329.63], 5.0, "A minor chord, melancholic"),
        ("f_major_chord", [174.61, 220.00, 261.63], 5.0, "F major chord, rich and full"),
        ("d_minor_chord", [146.83, 174.61, 220.00], 5.0, "D minor chord, dark and moody"),
        ("e_major_chord", [164.81, 207.65, 246.94], 5.0, "E major chord, bright and powerful"),
    ]
    
    print("Generating synthetic music samples...")
    for name, freqs, duration, desc in tqdm(music_samples, desc="Music samples"):
        audio = gen.generate_chord(freqs, duration, sr_music)
        audio = gen.apply_envelope(audio, attack=0.5, release=0.5)
        
        path = music_dir / f"{name}.wav"
        sf.write(path, audio, sr_music)
        
        music_data.append({
            "path": str(path),
            "duration": duration,
            "sample_rate": sr_music,
            "description": desc
        })
    
    sweep_samples = [
        ("bass_sweep", 80, 200, 5.0, "Bass frequency sweep, deep and rumbling"),
        ("mid_sweep", 200, 800, 5.0, "Mid frequency sweep, warm and present"),
        ("high_sweep", 800, 4000, 5.0, "High frequency sweep, bright and airy"),
    ]
    
    for name, f_start, f_end, duration, desc in sweep_samples:
        audio = gen.generate_sweep(f_start, f_end, duration, sr_music)
        audio = gen.apply_envelope(audio, attack=0.2, release=0.2)
        
        path = music_dir / f"{name}.wav"
        sf.write(path, audio, sr_music)
        
        music_data.append({
            "path": str(path),
            "duration": duration,
            "sample_rate": sr_music,
            "description": desc
        })
    
    noise_samples = [
        ("white_noise", "white", 5.0, "White noise, static and hissing"),
        ("pink_noise", "pink", 5.0, "Pink noise, soft and natural"),
        ("brown_noise", "brown", 5.0, "Brown noise, deep and rumbling"),
    ]
    
    print("Generating synthetic audio samples...")
    for name, noise_type, duration, desc in tqdm(noise_samples, desc="Noise samples"):
        audio = gen.generate_noise(duration, sr_audio, noise_type)
        audio = gen.apply_envelope(audio, attack=0.1, release=0.1)
        
        path = audio_dir / f"{name}.wav"
        sf.write(path, audio, sr_audio)
        
        audio_data.append({
            "path": str(path),
            "duration": duration,
            "sample_rate": sr_audio,
            "description": desc
        })
    
    tone_samples = [
        ("low_tone", 100, 3.0, "Low frequency tone, bass sound"),
        ("mid_tone", 440, 3.0, "Mid frequency tone, A440 reference"),
        ("high_tone", 2000, 3.0, "High frequency tone, treble sound"),
    ]
    
    for name, freq, duration, desc in tone_samples:
        audio = gen.generate_tone(freq, duration, sr_audio)
        audio = gen.apply_envelope(audio, attack=0.1, release=0.1)
        
        path = audio_dir / f"{name}.wav"
        sf.write(path, audio, sr_audio)
        
        audio_data.append({
            "path": str(path),
            "duration": duration,
            "sample_rate": sr_audio,
            "description": desc
        })
    
    return music_data, audio_data


def download_sample(url: str, output_path: Path, desc: str = "Downloading") -> bool:
    """Download a single audio sample."""
    if output_path.exists():
        return True
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def main():
    """Main function to download and generate audio samples."""
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    
    print("=" * 60)
    print("Free Audio Sample Collection")
    print("=" * 60)
    
    print("\n--- Generating Synthetic Audio ---")
    music_synth, audio_synth = generate_synthetic_dataset(processed_dir)
    print(f"Generated {len(music_synth)} synthetic music samples")
    print(f"Generated {len(audio_synth)} synthetic audio samples")
    
    summary = {
        "synthetic_music": len(music_synth),
        "synthetic_audio": len(audio_synth),
        "music_samples": music_synth,
        "audio_samples": audio_synth
    }
    
    summary_path = data_dir / "synthetic_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    
    return summary


if __name__ == "__main__":
    main()
