
"""Data loaders for AudioCraft datasets."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np

logger = logging.getLogger(__name__)


class AudioDataset(Dataset):
    """Base audio dataset class."""
    
    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 32000,
        duration: float = 10.0,
        file_extensions: List[str] = [".wav", ".mp3", ".flac"]
    ):
        """Initialize audio dataset.
        
        Args:
            data_dir: Directory containing audio files
            sample_rate: Target sample rate
            duration: Target duration in seconds
            file_extensions: Valid audio file extensions
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.max_samples = int(duration * sample_rate)
        
        # Find all audio files
        self.audio_files = []
        for ext in file_extensions:
            self.audio_files.extend(self.data_dir.rglob(f"*{ext}"))
        
        logger.info(f"Found {len(self.audio_files)} audio files in {data_dir}")
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and process audio file.
        
        Args:
            idx: Index of audio file
            
        Returns:
            Dictionary with audio tensor and metadata
        """
        audio_path = self.audio_files[idx]
        
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Ensure mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Trim or pad to target duration
            current_samples = waveform.shape[1]
            if current_samples > self.max_samples:
                # Random crop
                start = torch.randint(0, current_samples - self.max_samples, (1,)).item()
                waveform = waveform[:, start:start + self.max_samples]
            elif current_samples < self.max_samples:
                # Pad with zeros
                padding = self.max_samples - current_samples
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            return {
                "audio": waveform,
                "path": str(audio_path),
                "sample_rate": self.sample_rate
            }
        
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")
            # Return zeros on error
            return {
                "audio": torch.zeros(1, self.max_samples),
                "path": str(audio_path),
                "sample_rate": self.sample_rate
            }


class MusicDataset(AudioDataset):
    """Music-specific dataset with metadata."""
    
    def __init__(
        self,
        data_dir: str,
        metadata_file: Optional[str] = None,
        sample_rate: int = 32000,
        duration: float = 30.0
    ):
        """Initialize music dataset.
        
        Args:
            data_dir: Directory containing audio files
            metadata_file: Optional JSON/CSV file with descriptions
            sample_rate: Target sample rate
            duration: Target duration in seconds
        """
        super().__init__(data_dir, sample_rate, duration)
        self.metadata = {}
        
        if metadata_file:
            self._load_metadata(metadata_file)
    
    def _load_metadata(self, metadata_file: str) -> None:
        """Load metadata from file."""
        # Placeholder for metadata loading
        logger.info(f"Loading metadata from {metadata_file}")
        # self.metadata = json.load(open(metadata_file))
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item with metadata."""
        item = super().__getitem__(idx)
        
        # Add text description if available
        file_path = item["path"]
        if file_path in self.metadata:
            item["description"] = self.metadata[file_path]
        else:
            item["description"] = "music audio"
        
        return item


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """Create a DataLoader from dataset.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs
    )
