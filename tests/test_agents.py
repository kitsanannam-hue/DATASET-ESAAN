
"""Tests for AI agents."""

import pytest
import torch
from src.agents import BaseAgent, MusicGenAgent, AudioGenAgent


class TestMusicGenAgent:
    """Test MusicGen agent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        config = {"device": "cpu", "sample_rate": 32000}
        agent = MusicGenAgent(config)
        assert agent.sample_rate == 32000
        assert agent.device == "cpu"
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = {"device": "cpu"}
        agent = MusicGenAgent(config)
        assert agent.validate_config()


class TestAudioGenAgent:
    """Test AudioGen agent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        config = {"device": "cpu", "sample_rate": 16000}
        agent = AudioGenAgent(config)
        assert agent.sample_rate == 16000
        assert agent.device == "cpu"
