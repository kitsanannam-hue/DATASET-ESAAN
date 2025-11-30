
# AudioCraft AI Agents

Advanced autonomous AI agents for audio generation using Facebook Research's AudioCraft framework.

## Overview

This project implements high-accuracy AI agents for:
- **MusicGen**: Text-to-music generation
- **AudioGen**: Text-to-audio generation  
- **JASCO**: Joint audio and symbolic music generation with temporal control

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a simple demo
python scripts/demo.py
```

## Project Structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed directory organization.

## Agent Capabilities

Agents follow mission-oriented principles with:
- Expert-level reasoning and chain-of-thought
- Error-free execution with validation
- Efficient tool usage
- Structured, technical communication

See [AGENTS.md](AGENTS.md) for agent behavior details.

## Configuration

This project uses Hydra for configuration management. Configs are in `configs/`:

```bash
# Use default config
python src/train.py

# Override settings
python src/train.py dataset=custom model=musicgen_medium device=cuda
```

## Development

```bash
# Run tests
pytest tests/

# Code formatting
black src/

# Type checking
mypy src/
```

## Documentation

- [Project Structure](PROJECT_STRUCTURE.md)
- [Agent Behavior](AGENTS.md)
- [Data README](data/README.md)

## License

See LICENSE file.
