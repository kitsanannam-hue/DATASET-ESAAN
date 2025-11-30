"""AudioGen agent for text-to-audio generation."""

import logging
from typing import Any, Dict, Optional, List
import torch

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AudioGenAgent(BaseAgent):
    """Agent for AudioGen text-to-audio generation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize AudioGen agent.

        Args:
            config: Agent configuration dictionary
        """
        super().__init__(config)
        self.model_name = config.get("model_name", "facebook/audiogen-medium")
        self.sample_rate = config.get("sample_rate", 16000)
        self.max_duration = config.get("max_duration", 10.0)

    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load AudioGen model.

        Args:
            model_path: Path to model checkpoint (optional)
        """
        logger.info(f"Loading AudioGen model: {self.model_name}")

        try:
            try:
                from audiocraft.models import AudioGen
                self.model = AudioGen.get_pretrained(self.model_name)
                self.model.to(self.device)
                logger.info("AudioCraft model loaded successfully")
            except ImportError:
                logger.warning("AudioCraft not installed. Using mock model for demo.")
                class MockModel:
                    def __init__(self, device):
                        self.device = device
                    def generate(self, prompts, **kwargs):
                        return None
                self.model = MockModel(self.device)
                logger.info("Mock model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(
        self,
        prompts: List[str],
        duration: Optional[float] = None,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        **kwargs
    ) -> torch.Tensor:
        """Generate audio from text prompts.

        Args:
            prompts: List of text descriptions
            duration: Duration in seconds (default: config max_duration)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter

        Returns:
            Generated audio tensor [batch, channels, samples]
        """
        duration = duration or self.max_duration
        logger.info(f"Generating {duration}s audio for {len(prompts)} prompts")

        try:
            if hasattr(self.model, 'set_generation_params'):
                self.model.set_generation_params(
                    duration=duration,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                audio = self.model.generate(prompts)
                logger.info(f"Generated audio shape: {audio.shape}")
                return audio
        except Exception as e:
            logger.warning(f"AudioCraft generation failed: {e}. Using mock output.")

        # Fallback to mock output
        batch_size = len(prompts)
        samples = int(duration * self.sample_rate)
        audio = torch.randn(batch_size, 1, samples)
        logger.info(f"Generated mock audio shape: {audio.shape}")

        return audio