
"""Demo script for AudioCraft AI Agents."""

import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents import MusicGenAgent, AudioGenAgent
from evaluation import AudioMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_musicgen():
    """Demo MusicGen agent."""
    logger.info("=== MusicGen Demo ===")
    
    config = {
        "device": "cpu",
        "model_name": "facebook/musicgen-small",
        "sample_rate": 32000,
        "max_duration": 10.0
    }
    
    agent = MusicGenAgent(config)
    agent.load_model()
    
    prompts = [
        "upbeat electronic dance music",
        "calm acoustic guitar melody"
    ]
    
    audio = agent.generate(prompts, duration=5.0)
    logger.info(f"Generated audio shape: {audio.shape}")
    
    # Evaluate
    metrics = AudioMetrics.evaluate_generation(audio[0])
    logger.info(f"Metrics: {metrics}")


def demo_audiogen():
    """Demo AudioGen agent."""
    logger.info("=== AudioGen Demo ===")
    
    config = {
        "device": "cpu",
        "model_name": "facebook/audiogen-medium",
        "sample_rate": 16000,
        "max_duration": 5.0
    }
    
    agent = AudioGenAgent(config)
    agent.load_model()
    
    prompts = ["dog barking", "rain falling on leaves"]
    
    audio = agent.generate(prompts, duration=3.0)
    logger.info(f"Generated audio shape: {audio.shape}")


if __name__ == "__main__":
    demo_musicgen()
    demo_audiogen()
    logger.info("Demo complete!")
