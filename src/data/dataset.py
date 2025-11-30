#!/usr/bin/env python3
"""
AudioCraft Dataset Classes
Provides PyTorch Dataset implementations for loading audio data for training.
"""

import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np


class AudioCraftDataset(Dataset):
    """Base dataset class for AudioCraft training."""
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        sample_rate: int = 32000,
        duration: float = 30.0,
        return_info: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            manifest_path: Path to JSONL manifest file
            sample_rate: Target sample rate
            duration: Maximum audio duration in seconds
            return_info: Whether to return metadata with audio
        """
        self.manifest_path = Path(manifest_path)
        self.sample_rate = sample_rate
        self.duration = duration
        self.return_info = return_info
        self.max_samples = int(sample_rate * duration)
        
        self.data = self._load_manifest()
    
    def _load_manifest(self) -> List[Dict]:
        """Load manifest file."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        
        data = []
        with open(self.manifest_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        if Path(item['path']).exists():
                            data.append(item)
                    except json.JSONDecodeError:
                        continue
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _load_audio(self, path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        try:
            waveform, sr = torchaudio.load(path)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            if waveform.shape[1] > self.max_samples:
                waveform = waveform[:, :self.max_samples]
            elif waveform.shape[1] < self.max_samples:
                padding = self.max_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            return waveform.squeeze(0)
        
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(self.max_samples)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        item = self.data[idx]
        audio = self._load_audio(item['path'])
        
        if self.return_info:
            info = {
                'path': item['path'],
                'description': item.get('description', ''),
                'duration': item.get('duration', 0),
                'sample_rate': item.get('sample_rate', self.sample_rate)
            }
            return audio, info
        
        return audio


class MusicGenDataset(AudioCraftDataset):
    """Dataset for MusicGen training at 32kHz."""
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        duration: float = 30.0,
        return_info: bool = False
    ):
        super().__init__(
            manifest_path=manifest_path,
            sample_rate=32000,
            duration=duration,
            return_info=return_info
        )


class AudioGenDataset(AudioCraftDataset):
    """Dataset for AudioGen training at 16kHz."""
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        duration: float = 10.0,
        return_info: bool = False
    ):
        super().__init__(
            manifest_path=manifest_path,
            sample_rate=16000,
            duration=duration,
            return_info=return_info
        )


class TextConditionedDataset(Dataset):
    """Dataset with text conditioning for AudioCraft models."""
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        sample_rate: int = 32000,
        duration: float = 30.0,
        tokenizer=None
    ):
        self.manifest_path = Path(manifest_path)
        self.sample_rate = sample_rate
        self.duration = duration
        self.max_samples = int(sample_rate * duration)
        self.tokenizer = tokenizer
        
        self.data = self._load_manifest()
    
    def _load_manifest(self) -> List[Dict]:
        """Load manifest file."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        
        data = []
        with open(self.manifest_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        if Path(item['path']).exists():
                            data.append(item)
                    except json.JSONDecodeError:
                        continue
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _load_audio(self, path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        try:
            waveform, sr = torchaudio.load(path)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            if waveform.shape[1] > self.max_samples:
                waveform = waveform[:, :self.max_samples]
            elif waveform.shape[1] < self.max_samples:
                padding = self.max_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            return waveform.squeeze(0)
        
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(self.max_samples)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        audio = self._load_audio(item['path'])
        description = item.get('description', '')
        
        result = {
            'audio': audio,
            'description': description,
            'path': item['path']
        }
        
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                description,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=77
            )
            result['input_ids'] = tokens['input_ids'].squeeze(0)
            result['attention_mask'] = tokens['attention_mask'].squeeze(0)
        
        return result


def create_dataloader(
    manifest_path: str,
    dataset_type: str = "musicgen",
    batch_size: int = 4,
    num_workers: int = 2,
    shuffle: bool = True,
    return_info: bool = False
) -> DataLoader:
    """
    Create a DataLoader for AudioCraft training.
    
    Args:
        manifest_path: Path to JSONL manifest
        dataset_type: "musicgen" or "audiogen"
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        return_info: Whether to return metadata
    
    Returns:
        DataLoader instance
    """
    if dataset_type == "musicgen":
        dataset = MusicGenDataset(manifest_path, return_info=return_info)
    elif dataset_type == "audiogen":
        dataset = AudioGenDataset(manifest_path, return_info=return_info)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def verify_dataset(manifest_path: str, dataset_type: str = "musicgen") -> Dict:
    """
    Verify dataset integrity and return statistics.
    
    Args:
        manifest_path: Path to JSONL manifest
        dataset_type: "musicgen" or "audiogen"
    
    Returns:
        Dictionary with dataset statistics
    """
    dataloader = create_dataloader(
        manifest_path,
        dataset_type=dataset_type,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        return_info=True
    )
    
    stats = {
        "total_samples": len(dataloader.dataset),
        "valid_samples": 0,
        "invalid_samples": 0,
        "total_duration": 0.0,
        "descriptions": []
    }
    
    for batch in dataloader:
        audio, info = batch
        if audio.abs().max() > 0:
            stats["valid_samples"] += 1
            stats["total_duration"] += info["duration"][0].item() if isinstance(info["duration"], torch.Tensor) else info["duration"][0]
            if info["description"][0]:
                stats["descriptions"].append(info["description"][0])
        else:
            stats["invalid_samples"] += 1
    
    stats["unique_descriptions"] = len(set(stats["descriptions"]))
    del stats["descriptions"]
    
    return stats
