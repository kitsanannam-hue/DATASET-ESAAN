
"""Main training script for AudioCraft AI Agents."""

import logging
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents import MusicGenAgent, AudioGenAgent
from training import AudioCraftTrainer
from data.loaders import MusicDataset, create_dataloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function.
    
    Args:
        cfg: Hydra configuration
    """
    logger.info("=" * 60)
    logger.info("AudioCraft AI Agents Training")
    logger.info("=" * 60)
    logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seed
    torch.manual_seed(cfg.seed)
    
    # Setup device
    device = cfg.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Initialize agent
    agent_config = {
        "device": device,
        "model_name": cfg.model.name,
        "sample_rate": cfg.model.sample_rate,
        "max_duration": cfg.model.max_duration
    }
    
    logger.info(f"Initializing {cfg.model.type} agent...")
    if cfg.model.type == "musicgen":
        agent = MusicGenAgent(agent_config)
    elif cfg.model.type == "audiogen":
        agent = AudioGenAgent(agent_config)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    
    # Load model
    agent.load_model()
    
    # Prepare datasets
    logger.info("Loading datasets...")
    data_dir = Path(cfg.paths.data_dir) / "processed"
    
    if not data_dir.exists():
        logger.warning(f"Data directory {data_dir} does not exist. Creating mock data...")
        data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        train_dataset = MusicDataset(
            data_dir=str(data_dir),
            sample_rate=cfg.model.sample_rate,
            duration=cfg.model.max_duration
        )
        
        train_loader = create_dataloader(
            train_dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=True,
            num_workers=cfg.dataset.num_workers
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
    except Exception as e:
        logger.warning(f"Could not load data: {e}. Using mock training...")
        train_loader = None
    
    # Initialize trainer
    if agent.model is not None:
        trainer_config = {
            "learning_rate": cfg.solver.lr,
            "weight_decay": cfg.solver.weight_decay,
            "num_epochs": cfg.solver.epochs
        }
        
        trainer = AudioCraftTrainer(
            model=agent.model,
            config=trainer_config,
            device=device
        )
        
        # Training loop
        logger.info("Starting training...")
        for epoch in range(cfg.solver.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{cfg.solver.epochs}")
            
            if train_loader:
                metrics = trainer.train_epoch(train_loader, epoch)
                logger.info(f"Training metrics: {metrics}")
            
            # Save checkpoint
            checkpoint_path = Path(cfg.paths.models_dir) / f"checkpoint_epoch_{epoch}.pt"
            trainer.save_checkpoint(
                str(checkpoint_path),
                epoch,
                metrics if train_loader else {}
            )
    else:
        logger.warning("Model not loaded. Skipping training.")
    
    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
