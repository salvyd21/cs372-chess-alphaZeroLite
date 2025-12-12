# CS 372 Chess RL Agent: AlphaZero-Lite

## What it Does
This project trains a chess-playing AI agent using supervised learning on human games and reinforcement learning with MCTS (Monte Carlo Tree Search). The system implements an architecture adhering to the traditional AlphaZero style with:
- **Supervised pretraining** on Lichess master games
- **Neural network** with ResNet architecture (5 residual blocks, 128 channels)
- **Policy and value heads** for move prediction and position evaluation
- **MCTS integration** for enhanced move selection during gameplay
- **Self-play Evaluation** for speed-up of enhancing the model

## Quick Start

### Installation
```bash
# Clone the repo
git clone <repo-url>
cd cs372-chess-bot

# Install dependencies
pip install -r requirements.txt
```

### Train Supervised Model
```bash
# Train on preprocessed dataset
python main.py train --checkpoint supervised_best.pth

# Evaluate on test set
python main.py eval_test --checkpoint supervised_best.pth
```

### Play Against our Bot
```bash
# Interactive gameplay
python -m src.play
```

## Data Pipeline

### Raw Lichess PGN
We use the Lichess game database for supervised pretraining:
- **Source**: https://database.lichess.org/
- **File used**: `lichess_db_standard_rated_2025-01.pgn.zst`
- **Location**: `data/raw/lichess/`

Note: The raw PGN file is not committed to the repository due to size (~30 GiB).

### Processed Datasets

We convert PGN games (~50k) into tensors for representation:

- **Training set**: `data/processed/supervised_train.npz`
- **Validation set**: `data/processed/supervised_val.npz`
- **Test set**: `data/processed/supervised_test.npz`

**Format:**
- `X`: Board states as `(N, 12, 8, 8)` float32 tensors
  - 12 channels: 6 piece types × 2 colors (White P,N,B,R,Q,K + Black P,N,B,R,Q,K)
- `y`: Move indices as `(N,)` int64 array (range: 0-4671)

**Recreate datasets:**
```bash
python -m src.train_test.build_dataset
```

## Model Architecture

### ChessResNet
- **Input**: 12×8×8 board representation
- **Backbone**: 5 residual blocks with 128 channels
- **Policy Head**: Outputs probabilities over 4,672 possible actions
- **Value Head**: Outputs position evaluation in [-1, 1]
- **Action Space**: 73 action planes × 64 squares = 4,672 actions
  - 56 planes: sliding moves (8 directions × 7 distances)
  - 8 planes: knight moves
  - 9 planes: underpromotions (3 directions × 3 pieces)

## Evaluation

### Implemented Evaluation Scripts
Located in `src/eval/`:
- `testset_eval.py` - Top-1/Top-3 moves accuracy on held-out test set
- `against_random.py` - Play games vs random move opponent
- `against_greedy.py` - Play games vs greedy material baseline
- `against_stockfish.py` - Play games vs baseline open-source Stockfish engine
- `against_self.py` - Self-play evaluation (traditional to AlphaZero)
- `benchmark.py` - Performance benchmarking 
- `inspect_policy.py` - Visualize policy predictions
- `total_eval.py` - Comprehensive evaluation on all metrics

### Metrics
- Win-rate vs baselines (random, greedy, Stockfish depth-1)
- Top-K move prediction accuracy
- ELO estimate through tournament play
- Training/validation loss curves

## Video Links
- Demo Video:(https://youtu.be/FOa4rxtIB4I)
- Technical Walkthrough: TBA

## Attribution
See [attribution.md](attribution.md) for data sources, libraries, and external code credits.

## Individual Contributions
- **Partner Daniel**: Project setup, baseline models procurement, data collection, dataset processing, model architecture, Gym API environment framework, evaluation setup/framework/visualization, and repository organization
- **Partner Thatcher**: Model architecture, setting up neural network and wrapper, processing and transforming the data set, setting up and training the model, testing and evaluateing model, demo video, 
