# CS 372 Chess RL Agent: AlphaZero-Lite

## What it Does
This project trains a chess-playing AI using supervised learning on human games and reinforcement learning. 
Through additional levels of depth and comprehensive training/testing, this bot can play and beat up to XXXX ELO level player.

## Quick Start
<!-- - `conda env create -f environment.yml` OR `pip install -r requirements.txt`
- `python -m src.train_supervised`
- `python -m src.train_rl`
- `python -m src.play_human` --> tbd on these

## Raw Lichess PGN

We use a subset of the Lichess game database for supervised pretraining.

- Source: https://database.lichess.org/
- Example file used:
  - `lichess_db_standard_rated_2025-01.pgn`

Placed the downloaded PGN in:

- `data/raw/lichess/lichess_db_standard_rated_2025-01.pgn`

This file is **not committed** instead compressed as .zst to the repository due to original size (~30 GiB).

## Processed Data

We convert PGN -> (position, move) tensors and save them as:

- `data/processed/supervised_train.npz`
- Shape: `X: (N, 12, 8, 8)`; `y: (N,)`

You can recreate this file by running:

```bash
python -m src.train_test.supervised_pretrain --build-dataset
```

## Video Links
- Demo Video: TBA
- Technical Walkthrough: TBA

## Evaluation
- Baselines: random, greedy material, Stockfish depth-1
- Planned metrics: win-rate vs baselines, Elo estimate, learning curves

## Individual Contributions
- Partner Daniel: project setup and data collection
- Partner Thatcher: 
