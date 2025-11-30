
"""Base agent class for AudioCraft AI agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all AudioCraft AI agents."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize agent with configuration.
        
        Args:
            config: Agent configuration dictionary
        """
        self.config = config
        self.model = None
        self.device = config.get("device", "cpu")
        logger.info(f"Initializing {self.__class__.__name__} on {self.device}")
    
    @abstractmethod
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load the model.
        
        Args:
            model_path: Path to model checkpoint (optional)
        """
        pass
    
    @abstractmethod
    def generate(self, *args, **kwargs) -> Any:
        """Generate output based on input.
        
        Returns:
            Generated output
        """
        pass
    
    def validate_config(self) -> bool:
        """Validate agent configuration.
        
        Returns:
            True if config is valid
        """
        required_keys = ["device"]
        return all(key in self.config for key in required_keys)
