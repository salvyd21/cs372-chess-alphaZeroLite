import argparse
import sys
from pathlib import Path
import numpy as np
import chess
import chess.engine
import time

from chess_engine.ChessGame import ChessGame
from chess_nnet.NNetWrapper import NNetWrapper
from core.MCTS import MCTS
from chess_engine.action_encoding import decode_action, get_all_possible_moves, ACTION_SIZE
from chess_engine.state_encoding import board_to_tensor

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
        else: # Sample from probabilities
            action = np.random.choice(len(masked_probs), p=masked_probs)

        return decode_action(action, board)

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
        
        if isinstance(player, MCTSPlayer):
            move = player.play(board, temp=0)
        else: # player2 is Stockfish --> expects a python-chess board object
            move = player.play(board)
            
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
                
    return 0 # If loop finishes without explicit win/loss (should be covered by getGameEnded)


def main():
    parser = argparse.ArgumentParser(description="Play MCTS-NN agent against Stockfish.")
    parser.add_argument('--stockfish_path', type=str, required=True,
                        help='Path to the Stockfish executable.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained MCTS-NN model (.pth file).')
    parser.add_argument('--num_games', type=int, default=1, help='Number of games to play.')
    parser.add_argument('--mcts_sims', type=int, default=50, help='Number of MCTS simulations per move.')
    parser.add_argument('--cpuct', type=float, default=1.0, help='CPUCT parameter for MCTS.')
    parser.add_argument('--display_board', action='store_true', help='Display board during play.')
    parser.add_argument('--stockfish_skill_level', type=int, default=10, 
                        help='Stockfish skill level (0-20).')
    parser.add_argument('--stockfish_time_limit', type=float, default=0.1, 
                        help='Stockfish time limit per move in seconds.')

    args = parser.parse_args()

    # 1. Initialize Game and MCTS-NN Player
    print("Initializing MCTS-NN player...")
    game = ChessGame()
    nnet = NNetWrapper(game)
    nnet.load_checkpoint(Path(args.model_path).parent, Path(args.model_path).name)
    
    mcts_args = type('MCTSArgs', (), {'numMCTSSims': args.mcts_sims, 'cpuct': args.cpuct})()
    mcts_player = MCTSPlayer(game, nnet, mcts_args)
    print("MCTS-NN player initialized.")

    # 2. Initialize Stockfish Engine
    print(f"Initializing Stockfish engine at {args.stockfish_path}...")
    try:
        engine = chess.engine.popen_uci(args.stockfish_path)
        engine.configure({"UCI_LimitStrength": False, "Skill Level": args.stockfish_skill_level})

        class StockfishPlayer:
            def __init__(self, engine, time_limit):
                self.engine = engine
                self.time_limit = time_limit

            def play(self, board):
                # Use a specific time limit for Stockfish
                result = self.engine.play(board, chess.engine.Limit(time=self.time_limit))
                return result.move

        stockfish_player = StockfishPlayer(engine, args.stockfish_time_limit)
        print("Stockfish engine initialized.")

    except Exception as e:
        print(f"Error initializing Stockfish: {e}")
        print("Please ensure the Stockfish executable path is correct and it's a valid UCI engine.")
        sys.exit(1)

    # 3. Play Games
    print(f"\nStarting {args.num_games} games against Stockfish...")
    mcts_wins = 0
    stockfish_wins = 0
    draws = 0

    for i in range(args.num_games):
        print(f"\n--- Game {i+1}/{args.num_games} ---")
        # Alternate who plays white
        if i % 2 == 0:
            # MCTS plays white, Stockfish plays black
            print("MCTS-NN (White) vs Stockfish (Black)")
            result = play_game(mcts_player, stockfish_player, display=args.display_board)
        else:
            # Stockfish plays white, MCTS plays black
            print("Stockfish (White) vs MCTS-NN (Black)")
            # For this scenario, we need a slight modification to play_game or have players swap logic
            # For simplicity, let's keep MCTS as player1 and handle its turn, and Stockfish as player2
            # The `play_game` function already handles player switching internally based on `current_player_idx`.
            # We need to swap the actual player objects for the `play_game` call.
            result = play_game(stockfish_player, mcts_player, display=args.display_board)
            # Invert result from stockfish perspective to MCTS perspective
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

    # 4. Clean up Stockfish engine
    try:
        engine.quit()
    except Exception as e:
        print(f"Error quitting Stockfish engine: {e}")

if __name__ == '__main__':
    main()
```
