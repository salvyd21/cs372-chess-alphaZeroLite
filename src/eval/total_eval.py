# run python src/eval/tournament_eval.py --model_dir " " --opponent_types stockfish greedy random --stockfish_path " " --output_dir data/eval
# in terminal to generate this file and execute evaluation on all model types
import argparse
import sys
from pathlib import Path
import numpy as np
import chess
import chess.engine
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.chess_engine.ChessGame import ChessGame
from src.chess_nnet.NNetWrapper import NNetWrapper
from src.core.MCTS import MCTS
from src.chess_engine.action_encoding import decode_action, ACTION_SIZE
from src.chess_engine.state_encoding import board_to_tensor

# Placeholder for MCTSPlayer and StockfishPlayer classes
# Assume evaluate_model will handle player initialization internally or accept player objects.

def evaluate_model(model_path, opponent_type, num_games, mcts_sims, cpuct, stockfish_path=None, stockfish_skill_level=None, stockfish_time_limit=None):
    """
    Evaluates a given model checkpoint against a specified opponent type.
    Returns a dictionary of results (e.g., wins, losses, draws).
    """
    print(f"Evaluating model {model_path} against {opponent_type} for {num_games} games...")
    # This function will eventually call a game-playing logic, similar to against_stockfish.py
    # For now, return dummy results
    return {"model": model_path.name, "opponent": opponent_type, "wins": 0, "losses": 0, "draws": 0}

def plot_results(results_df, output_dir):
    """
    Generates and saves plots based on the evaluation results dataframe.
    """
    print("Generating plots...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Example: Plotting win rates
    # This is a placeholder; actual plotting logic will depend on the `results_df` structure
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='wins', hue='opponent', data=results_df)
    plt.title('Model Performance Against Opponents')
    plt.ylabel('Wins')
    plt.xlabel('Model Checkpoint')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "win_rates.png")
    plt.close()
    print(f"Plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Orchestrate MCTS-NN model evaluations against various opponents.")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to a directory containing model checkpoints or a single .pth model file.')
    parser.add_argument('--opponent_types', nargs='+', default=['stockfish'],
                        help='List of opponent types (e.g., \'random\', \'greedy\', \'stockfish\').')
    parser.add_argument('--num_games', type=int, default=10,
                        help='Number of games to play per evaluation.')
    parser.add_argument('--mcts_sims', type=int, default=50,
                        help='Number of MCTS simulations for the agent per move.')
    parser.add_argument('--cpuct', type=float, default=1.0,
                        help='CPUCT parameter for MCTS.')
    parser.add_argument('--stockfish_path', type=str, default=None,
                        help='Path to the Stockfish executable (required if \'stockfish\' is an opponent).')
    parser.add_argument('--stockfish_skill_level', type=int, default=10,
                        help='Stockfish skill level (0-20).')
    parser.add_argument('--stockfish_time_limit', type=float, default=0.1,
                        help='Stockfish time limit per move in seconds.')
    parser.add_argument('--output_dir', type=str, default='data/eval',
                        help='Directory to save results and plots.')

    args = parser.parse_args()

    model_dir_path = Path(args.model_dir)
    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    model_paths = []
    if model_dir_path.is_file() and model_dir_path.suffix == '.pth.tar':
        model_paths.append(model_dir_path)
    elif model_dir_path.is_dir():
        model_paths.extend(sorted(list(model_dir_path.glob('*.pth.tar'))))
    else:
        print(f"Error: model_dir '{args.model_dir}' is neither a .pth.tar file nor a directory.")
        sys.exit(1)

    if not model_paths:
        print(f"No .pth.tar model files found in '{args.model_dir}'. Exiting.")
        sys.exit(1)

    print(f"Found {len(model_paths)} models to evaluate.")
    all_results = []

    for model_path in model_paths:
        for opponent_type in args.opponent_types:
            if opponent_type == 'stockfish' and not args.stockfish_path:
                print("Error: --stockfish_path is required when 'stockfish' is an opponent type.")
                sys.exit(1)

            results = evaluate_model(
                model_path,
                opponent_type,
                args.num_games,
                args.mcts_sims,
                args.cpuct,
                args.stockfish_path,
                args.stockfish_skill_level,
                args.stockfish_time_limit
            )
            all_results.append(results)
