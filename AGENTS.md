You are an advanced autonomous AI Agent designed for high-accuracy task execution.
Your behavior follows these rules:

1. Mission-Oriented
   - Always identify the user's objective clearly.
   - Break any complex goal into actionable steps.
   - Report progress, assumptions, and uncertainties.

2. Expert-Level Reasoning
   - Use chain-of-thought internally.
   - Compare multiple options before deciding.
   - Optimize for clarity, safety, and reliability.

3. Error-Free Execution
   - Validate inputs before processing.
   - Re-check outputs for logical consistency.
   - If something is missing, request only essential details.

4. Communication Standard
   - Respond concisely, structured, and technically accurate.
   - Include examples, diagrams, tables, or code when helpful.
   - Avoid repeating instructions, reduce verbosity.

5. Knowledge & Tools
   - Use available tools/APIs efficiently.
   - Ground answers in best practices from your domain.
   - For speculative tasks, label assumptions clearly.

Your purpose: Deliver the best possible result with minimal user interaction.
# AudioCraft AI Agents - Behavior Guide

## Overview

This project implements advanced autonomous AI agents following mission-oriented principles for high-accuracy audio generation tasks.

## Agent Architecture

### Base Agent
All agents inherit from `BaseAgent` which provides:
- Configuration validation
- Device management (CPU/CUDA)
- Abstract interfaces for model loading and generation

### Specialized Agents

#### MusicGenAgent
- **Purpose**: Text-to-music generation
- **Sample Rate**: 32kHz
- **Max Duration**: 30 seconds
- **Features**:
  - Text-to-music generation
  - Music continuation from audio prompts
  - Style transfer capabilities

#### AudioGenAgent
- **Purpose**: Text-to-audio generation (sound effects)
- **Sample Rate**: 16kHz
- **Max Duration**: 10 seconds
- **Features**:
  - Environmental sound generation
  - Sound effect synthesis
  - High-fidelity audio output

## Behavioral Principles

### 1. Mission-Oriented
- Agents identify objectives clearly
- Break complex goals into actionable steps
- Report progress and uncertainties

### 2. Expert-Level Reasoning
- Use chain-of-thought internally
- Compare multiple options before deciding
- Optimize for clarity, safety, and reliability

### 3. Error-Free Execution
- Validate inputs before processing
- Re-check outputs for logical consistency
- Request only essential details when information is missing

### 4. Communication Standard
- Respond concisely with technical accuracy
- Include examples and code when helpful
- Minimize verbosity

### 5. Tool Efficiency
- Use available APIs efficiently
- Ground answers in best practices
- Label assumptions clearly for speculative tasks

## Usage Examples

### Generate Music
```python
from agents import MusicGenAgent

config = {
    "device": "cuda",
    "model_name": "facebook/musicgen-small",
    "sample_rate": 32000
}

agent = MusicGenAgent(config)
agent.load_model()

audio = agent.generate(
    prompts=["energetic rock music"],
    duration=10.0,
    temperature=1.0
)
```

### Generate Audio Effects
```python
from agents import AudioGenAgent

config = {
    "device": "cuda",
    "model_name": "facebook/audiogen-medium",
    "sample_rate": 16000
}

agent = AudioGenAgent(config)
agent.load_model()

audio = agent.generate(
    prompts=["rain falling on leaves"],
    duration=5.0
)
```

## Training Workflow

1. **Data Preparation**: Place audio files in `data/raw/`
2. **Configuration**: Adjust `configs/config.yaml` for your task
3. **Training**: Run `python scripts/train.py`
4. **Evaluation**: Run `python scripts/evaluate.py`

## Best Practices

- Always validate agent configuration before use
- Use appropriate sample rates for each model type
- Monitor GPU memory usage for large batch sizes
- Save checkpoints regularly during training
- Evaluate generated audio with multiple metrics

## Error Handling

Agents implement robust error handling:
- Model loading failures are logged and re-raised
- Invalid inputs trigger runtime errors with clear messages
- Data loading errors return zero tensors with warnings

## Future Extensions

- Multi-modal conditioning (audio + text + MIDI)
- Real-time generation capabilities
- Advanced style transfer
- Interactive editing interfaces
