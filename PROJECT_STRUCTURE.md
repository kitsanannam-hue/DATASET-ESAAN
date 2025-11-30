
# AudioCraft AI Agents - Project Structure

## Root Directory Structure

```
audiocraft-ai-agents/
├── data/                          # Dataset storage and processing
│   ├── raw/                       # Original, immutable data
│   │   ├── audio/                 # Raw audio files
│   │   ├── metadata/              # Original metadata files
│   │   └── annotations/           # Manual annotations
│   ├── processed/                 # Cleaned, processed data
│   │   ├── audio/                 # Processed audio (normalized, resampled)
│   │   ├── features/              # Extracted features (spectrograms, embeddings)
│   │   └── manifests/             # Data manifests (.jsonl, .jsonl.gz)
│   ├── external/                  # External datasets or references
│   ├── interim/                   # Intermediate processing results
│   └── cache/                     # Cached embeddings, chroma, etc.
│
├── models/                        # Trained models and checkpoints
│   ├── compression/               # EnCodec models
│   ├── lm/                        # Language models (MusicGen, AudioGen)
│   ├── jasco/                     # JASCO models
│   ├── pretrained/                # Pre-trained model weights
│   └── checkpoints/               # Training checkpoints
│
├── configs/                       # Configuration files
│   ├── dataset/                   # Dataset-specific configs
│   ├── model/                     # Model architecture configs
│   ├── solver/                    # Training/evaluation configs
│   ├── conditioner/               # Conditioning configs
│   └── experiments/               # Experiment-specific configs
│
├── src/                           # Source code
│   ├── agents/                    # AI agent implementations
│   │   ├── __init__.py
│   │   ├── base_agent.py         # Base agent class
│   │   ├── music_agent.py        # Music generation agent
│   │   ├── audio_agent.py        # Audio generation agent
│   │   └── jasco_agent.py        # JASCO agent
│   ├── data/                      # Data processing modules
│   │   ├── __init__.py
│   │   ├── loaders.py            # Data loaders
│   │   ├── processors.py         # Data processors
│   │   └── augmentation.py       # Data augmentation
│   ├── models/                    # Model implementations
│   │   ├── __init__.py
│   │   └── custom_models.py      # Custom model architectures
│   ├── training/                  # Training utilities
│   │   ├── __init__.py
│   │   ├── trainers.py
│   │   └── callbacks.py
│   ├── evaluation/                # Evaluation utilities
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── evaluators.py
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       ├── audio_utils.py
│       └── helpers.py
│
├── notebooks/                     # Jupyter notebooks
│   ├── exploratory/               # Data exploration
│   ├── experiments/               # Experimental notebooks
│   └── demos/                     # Demo notebooks
│
├── scripts/                       # Standalone scripts
│   ├── preprocessing/             # Data preprocessing scripts
│   ├── training/                  # Training scripts
│   ├── evaluation/                # Evaluation scripts
│   └── deployment/                # Deployment scripts
│
├── tests/                         # Unit and integration tests
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── docs/                          # Documentation
│   ├── api/                       # API documentation
│   ├── guides/                    # User guides
│   └── architecture/              # Architecture documentation
│
├── outputs/                       # Generated outputs
│   ├── audio/                     # Generated audio files
│   ├── visualizations/            # Plots, spectrograms
│   ├── logs/                      # Training logs
│   └── reports/                   # Evaluation reports
│
├── experiments/                   # Experiment tracking (dora)
│   └── grids/                     # Dora grid configurations
│
├── assets/                        # Static assets
│   ├── chord_mappings/            # Chord to index mappings
│   ├── pretrained_weights/        # Downloaded pretrained weights
│   └── reference_audio/           # Reference audio samples
│
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
└── AGENTS.md                      # Agent behavior documentation
```

## Key Directory Purposes

### `/data/`
- **raw/**: Store original, immutable datasets
- **processed/**: Store cleaned, normalized data ready for training
- **manifests/**: JSON/JSONL files describing dataset structure
- **cache/**: Store computed embeddings to avoid recomputation

### `/models/`
Organize by model type following AudioCraft conventions:
- compression (EnCodec models)
- lm (language models)
- Separate pretrained from fine-tuned models

### `/configs/`
YAML configuration files using Hydra/Dora patterns:
- Modular configs that can be composed
- Environment-specific overrides
- Experiment configurations

### `/src/`
Production-quality source code:
- Modular, reusable components
- Clear separation of concerns
- Well-documented APIs

### `/notebooks/`
Interactive development and analysis:
- exploratory/: Data exploration and analysis
- experiments/: Quick prototyping
- demos/: Demonstration notebooks (like jasco_demo.ipynb)

### `/outputs/`
All generated artifacts (excluded from version control):
- Generated audio samples
- Training logs and metrics
- Evaluation results

## Dataset Organization Best Practices

### Audio Files
```
data/processed/audio/
├── music/
│   ├── genre_a/
│   │   ├── track001.wav
│   │   └── track001.json  # metadata
│   └── genre_b/
├── sound_effects/
└── speech/
```

### Manifests
```
data/processed/manifests/
├── train_data.jsonl
├── valid_data.jsonl
├── test_data.jsonl
└── metadata/
    └── dataset_stats.json
```

### Cached Features
```
data/cache/
├── chroma_stem/
├── clap_embeddings/
├── t5_embeddings/
└── melody_salience/
```

## Configuration Pattern

```yaml
# configs/dataset/my_dataset.yaml
name: my_audio_dataset
sample_rate: 32000
channels: 2
segment_duration: 30
batch_size: 16
train:
  path: data/processed/manifests/train_data.jsonl
  num_samples: 10000
valid:
  path: data/processed/manifests/valid_data.jsonl
  num_samples: 1000
```

## Model Checkpoint Organization

```
models/
├── musicgen/
│   ├── small/
│   │   ├── checkpoint_epoch_100.pt
│   │   └── best_model.pt
│   └── medium/
├── audiogen/
└── jasco/
    ├── chords_drums_400M/
    └── chords_drums_1B/
```

## Naming Conventions

1. **Files**: lowercase_with_underscores.py
2. **Classes**: PascalCase
3. **Functions**: snake_case
4. **Constants**: UPPER_CASE
5. **Audio files**: descriptive_name_samplerate_channels.wav
6. **Manifests**: {split}_data.jsonl (train_data.jsonl, valid_data.jsonl)

## Version Control

`.gitignore` should exclude:
```
data/raw/*
data/processed/*
data/cache/*
models/checkpoints/*
outputs/*
*.pyc
__pycache__/
.ipynb_checkpoints/
```

Track with version control:
- Source code
- Configuration files
- Documentation
- Small reference files
- Model architecture definitions

## Notes

- Follow AudioCraft patterns for compatibility
- Use Hydra/Dora for configuration management
- Maintain clear separation between code and data
- Document data provenance and transformations
- Use consistent naming across the project
```
