#!/usr/bin/env bash
set -euo pipefail

echo "Building freq_dataset"
python "helpers/build_freq_dict.py"

echo "Setup complete."
