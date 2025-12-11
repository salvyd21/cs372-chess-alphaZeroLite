import argparse
import sys
import os
from pathlib import Path
import numpy as np
import chess
import chess.engine
import time
import torch
import torch.optim as optim

# Ensure src is in python path
project_root = '/content/cs372-chess-alphaZeroLite'
if project_root not in sys.path:
    sys.path.append(os.path.join(project_root, 'src'))

from chess_engine.ChessGame import ChessGame
from chess_nnet.NNetWrapper import NNetWrapper
from chess_nnet.ChessNNet import ChessResNet
from core.MCTS import MCTS
from chess_engine.action_encoding import decode_move_index, ACTION_SIZE
from chess_engine.state_encoding import board_to_tensor

# Helper class for unpickling legacy checkpoints
class Args:
    pass

class MCTSPlayer:
    def __init__(self, game, nnet, args):
        self.mcts = MCTS(game, nnet, args)
        self.game = game
        self.args = args

    def play(self, board, temp=0):
        canonical_board = self.game.getCanonicalForm(board, board.turn == chess.WHITE)
        probs = self.mcts.getActionProb(canonical_board, temp=temp)

        # Filter out illegal moves based on the current board state
        valid_moves_mask = self.game.getValidMoves(board, board.turn == chess.WHITE)

        # Apply mask to probabilities and re-normalize
        masked_probs = probs * valid_moves_mask
        if np.sum(masked_probs) == 0: # If all valid moves have zero prob --> choose a valid move uniformly
            print("MCTS returned zero probability for all valid moves. Choosing uniformly.")
            masked_probs = valid_moves_mask / np.sum(valid_moves_mask)
        else:
            masked_probs /= np.sum(masked_probs)

        if temp == 0: # Choose best move deterministically
            action = np.argmax(masked_probs)
        else:
            action = np.random.choice(len(masked_probs), p=masked_probs)

        return decode_move_index(board, action)

def play_game(player1, player2, display=False):
    game = ChessGame()
    board = game.getInitBoard()
    current_player_idx = 0 # 0 for player1 (white), 1 for player2 (black)
    players = [player1, player2]

    history = [board.copy()]

    while game.getGameEnded(board, 1) == 0:
        if display:
            print(f"\n{'White' if board.turn == chess.WHITE else 'Black'}'s turn:")
            print(board)

        player = players[current_player_idx]

        try:
            if isinstance(player, MCTSPlayer):
                move = player.play(board, temp=0)
            else: # player2 is Stockfish
                move = player.play(board)
        except Exception as e:
            print(f"Error during play: {e}")
            break

        if move is None:
            print("Player returned no move, ending game.")
            break

        board.push(move)
        history.append(board.copy())

        current_player_idx = 1 - current_player_idx # Switch players

        # Check for game end conditions after each move
        game_result = game.getGameEnded(board, 1)
        if game_result != 0:
            if display:
                print("\nGame Over!")
                print(board)
            if game_result == 1:
                return 1 # MCTS wins
            elif game_result == -1:
                return -1 # Stockfish Wins
            else:
                return 0 # Draw

    return 0

def main():
    parser = argparse.ArgumentParser(description="Play MCTS-NN agent against Stockfish.")
    parser.add_argument('--stockfish_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_games', type=int, default=1)
    parser.add_argument('--mcts_sims', type=int, default=50)
    parser.add_argument('--cpuct', type=float, default=1.0)
    parser.add_argument('--display_board', action='store_true')
    parser.add_argument('--stockfish_skill_level', type=int, default=10)
    parser.add_argument('--stockfish_time_limit', type=float, default=0.1)

    args = parser.parse_args()

    # 1. Load Checkpoint Args
    print("Loading checkpoint config...")
    nnet_args = None
    try:
        # Load checkpoint to get args using weights_only=False to allow Args class if needed
        try:
            checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(args.model_path, map_location='cpu')

        training_args = checkpoint.get('args', {})

        # Helper class to wrap dict as object if needed (standard for AlphaZero repos)
        class Config:
            def __init__(self, dictionary):
                for k, v in dictionary.items():
                    setattr(self, k, v)
            def __getattr__(self, item):
                return None

        if isinstance(training_args, dict):
            nnet_args = Config(training_args)
        else:
            nnet_args = training_args

        # Use safe getattr for logging
        nc = getattr(nnet_args, 'num_channels', 'default')
        nrb = getattr(nnet_args, 'num_res_blocks', 'default')
        print(f"Checkpoint config: num_channels={nc} num_res_blocks={nrb}")

    except Exception as e:
        print(f"Failed to load config from checkpoint: {e}")
        print("Falling back to default args (might cause architecture mismatch)")
        nnet_args = None

    # 2. Initialize Game and NNetWrapper (default)
    game = ChessGame()
    nnet = NNetWrapper(game)
    
    # 3. Update NNet architecture if args found in checkpoint
    if nnet_args:
        print("Updating NNet architecture from checkpoint args...")
        # Update internal args dict of the wrapper
        if getattr(nnet_args, 'num_channels', None): 
            nnet.args['num_channels'] = nnet_args.num_channels
        if getattr(nnet_args, 'num_res_blocks', None): 
            nnet.args['num_res_blocks'] = nnet_args.num_res_blocks
        if getattr(nnet_args, 'dropout', None): 
            nnet.args['dropout'] = nnet_args.dropout
        
        # Re-initialize the internal PyTorch model with new args
        nnet.nnet = ChessResNet(game, nnet.args).to(nnet.device)
        # Re-initialize optimizer (though not needed for inference, good practice)
        nnet.optimizer = optim.Adam(nnet.nnet.parameters(), lr=nnet.args['lr'])

    # 4. Load Weights
    nnet.load_checkpoint(Path(args.model_path).parent, Path(args.model_path).name)

    mcts_args = type('MCTSArgs', (), {'numMCTSSims': args.mcts_sims, 'cpuct': args.cpuct})()
    mcts_player = MCTSPlayer(game, nnet, mcts_args)
    print("MCTS-NN player initialized.")

    # 5. Initialize Stockfish Engine
    print(f"Initializing Stockfish engine at {args.stockfish_path}...")
    try:
        # Synchronous Engine
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
        engine.configure({"UCI_LimitStrength": False, "Skill Level": args.stockfish_skill_level})

        class StockfishPlayer:
            def __init__(self, engine, time_limit):
                self.engine = engine
                self.time_limit = time_limit

            def play(self, board):
                result = self.engine.play(board, chess.engine.Limit(time=self.time_limit))
                return result.move

        stockfish_player = StockfishPlayer(engine, args.stockfish_time_limit)
        print("Stockfish engine initialized.")

    except Exception as e:
        print(f"Error initializing Stockfish: {e}")
        sys.exit(1)

    # 6. Play Games
    print(f"\nStarting {args.num_games} games against Stockfish...")
    mcts_wins = 0
    stockfish_wins = 0
    draws = 0

    for i in range(args.num_games):
        print(f"\n--- Game {i+1}/{args.num_games} ---")
        if i % 2 == 0:
            print("MCTS-NN (White) vs Stockfish (Black)")
            result = play_game(mcts_player, stockfish_player, display=args.display_board)
        else:
            print("Stockfish (White) vs MCTS-NN (Black)")
            result = play_game(stockfish_player, mcts_player, display=args.display_board)
            result = -result

        if result == 1:
            mcts_wins += 1
            print("Result: MCTS-NN wins!")
        elif result == -1:
            stockfish_wins += 1
            print("Result: Stockfish wins!")
        else:
            draws += 1
            print("Result: Draw!")

    print("\n--- Tournament Results ---")
    print(f"Games played: {args.num_games}")
    print(f"MCTS-NN Wins: {mcts_wins}")
    print(f"Stockfish Wins: {stockfish_wins}")
    print(f"Draws: {draws}")

    engine.quit()

if __name__ == '__main__':
    main()
