
"""Evaluation metrics for audio generation."""

import logging
from typing import Dict, Any
import torch
import numpy as np

logger = logging.getLogger(__name__)


class AudioMetrics:
    """Audio quality and similarity metrics."""
    
    @staticmethod
    def signal_to_noise_ratio(
        target: torch.Tensor,
        prediction: torch.Tensor,
        eps: float = 1e-8
    ) -> float:
        """Calculate Signal-to-Noise Ratio.
        
        Args:
            target: Target audio tensor
            prediction: Predicted audio tensor
            eps: Small value to avoid division by zero
            
        Returns:
            SNR in dB
        """
        noise = target - prediction
        signal_power = torch.mean(target ** 2)
        noise_power = torch.mean(noise ** 2)
        
        snr = 10 * torch.log10(signal_power / (noise_power + eps))
        return snr.item()
    
    @staticmethod
    def mean_absolute_error(
        target: torch.Tensor,
        prediction: torch.Tensor
    ) -> float:
        """Calculate Mean Absolute Error.
        
        Args:
            target: Target audio tensor
            prediction: Predicted audio tensor
            
        Returns:
            MAE value
        """
        mae = torch.mean(torch.abs(target - prediction))
        return mae.item()
    
    @staticmethod
    def spectral_convergence(
        target: torch.Tensor,
        prediction: torch.Tensor
    ) -> float:
        """Calculate spectral convergence metric.
        
        Args:
            target: Target audio tensor
            prediction: Predicted audio tensor
            
        Returns:
            Spectral convergence value
        """
        # Placeholder - requires STFT implementation
        logger.info("Calculating spectral convergence")
        return 0.0
    
    @staticmethod
    def evaluate_generation(
        generated_audio: torch.Tensor,
        reference_audio: torch.Tensor = None
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of generated audio.
        
        Args:
            generated_audio: Generated audio tensor
            reference_audio: Optional reference audio for comparison
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            "duration": generated_audio.shape[-1] / 32000.0,  # assuming 32kHz
            "channels": generated_audio.shape[1] if len(generated_audio.shape) > 2 else 1,
            "peak_amplitude": torch.max(torch.abs(generated_audio)).item(),
            "rms_energy": torch.sqrt(torch.mean(generated_audio ** 2)).item()
        }
        
        if reference_audio is not None:
            metrics["snr"] = AudioMetrics.signal_to_noise_ratio(
                reference_audio, generated_audio
            )
            metrics["mae"] = AudioMetrics.mean_absolute_error(
                reference_audio, generated_audio
            )
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
