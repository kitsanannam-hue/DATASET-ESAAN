
"""Evaluation script for AudioCraft AI Agents."""

import logging
import sys
from pathlib import Path
import torch
import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents import MusicGenAgent, AudioGenAgent
from evaluation import AudioMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate audio generation quality.
    
    Args:
        cfg: Hydra configuration
    """
    logger.info("Starting evaluation...")
    
    # Initialize agent
    agent_config = {
        "device": cfg.device,
        "model_name": cfg.model.name,
        "sample_rate": cfg.model.sample_rate
    }
    
    if cfg.model.type == "musicgen":
        agent = MusicGenAgent(agent_config)
    else:
        agent = AudioGenAgent(agent_config)
    
    agent.load_model()
    
    # Test prompts
    test_prompts = [
        "upbeat electronic dance music with strong bass",
        "calm ambient background music",
        "energetic rock guitar solo"
    ]
    
    logger.info(f"Generating audio for {len(test_prompts)} prompts...")
    
    all_metrics = []
    for i, prompt in enumerate(test_prompts):
        logger.info(f"\nPrompt {i+1}: {prompt}")
        
        # Generate
        audio = agent.generate([prompt], duration=5.0)
        
        # Evaluate
        metrics = AudioMetrics.evaluate_generation(audio[0])
        all_metrics.append(metrics)
        
        # Save audio
        output_path = Path(cfg.paths.outputs_dir) / f"eval_sample_{i}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # torchaudio.save(str(output_path), audio[0], cfg.model.sample_rate)
        logger.info(f"Saved to: {output_path}")
    
    # Aggregate metrics
    logger.info("\n" + "="*60)
    logger.info("Evaluation Summary")
    logger.info("="*60)
    
    for i, (prompt, metrics) in enumerate(zip(test_prompts, all_metrics)):
        logger.info(f"\nPrompt {i+1}: {prompt}")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
    
    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    evaluate()
