# Setup Instructions

Clone the repo:
   ```bash
   git clone git@gitlab.oit.duke.edu:netid/cs372-chess-bot.git
   cd cs372-chess-bot

## Create environment:
    ```bash
    conda create -n chess-rl python=3.11
    conda activate chess-rl
    pip install -r requirements.txt
    (Optional) Install system packages for CUDA, etc.

## Download training data:

Place PGN files into data/raw/lichess/

Or run python -m scripts.download_data (later)

## Quick test:
    ```bash
    pytest
    python -m src.scripts.sanity_check