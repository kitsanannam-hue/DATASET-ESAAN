
"""Data loading utilities for AudioCraft datasets."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class AudioDataLoader:
    """Load and manage AudioCraft datasets."""
    
    def __init__(self, manifest_path: str, sample_rate: int = 32000):
        """Initialize data loader.
        
        Args:
            manifest_path: Path to JSONL manifest file
            sample_rate: Target sample rate
        """
        self.manifest_path = Path(manifest_path)
        self.sample_rate = sample_rate
        self.samples = []
        
        if self.manifest_path.exists():
            self.load_manifest()
    
    def load_manifest(self) -> List[Dict[str, Any]]:
        """Load samples from JSONL manifest.
        
        Returns:
            List of sample dictionaries
        """
        logger.info(f"Loading manifest from {self.manifest_path}")
        self.samples = []
        
        with open(self.manifest_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.samples.append(sample)
        
        logger.info(f"Loaded {len(self.samples)} samples")
        return self.samples
    
    def get_sample(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Sample dictionary or None
        """
        if 0 <= idx < len(self.samples):
            return self.samples[idx]
        return None
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
