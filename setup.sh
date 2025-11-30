
#!/bin/bash

# Install packages not available in Nix using pip with --user flag
python3 -m pip install --user hydra-core omegaconf wandb

echo "Setup complete!"
