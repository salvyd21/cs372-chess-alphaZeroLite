# Interactive chess gameplay against the trained AI.
# Allows human players to play against the MCTS-enhanced neural network.

import argparse
import chess
import numpy as np

from chess_engine.ChessGame import ChessGame
from chess_nnet.NNetWrapper import NNetWrapper
from chess_engine.action_encoding import encode_move_index, decode_move_index
from core.MCTS import MCTS


def display_board(board: chess.Board):
    """Display the chess board in a readable format."""
    print("\n" + "-" * 40)
    print(board)
    print("-" * 40)
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    print(f"FEN: {board.fen()}")
    print()


def get_human_move(board: chess.Board) -> chess.Move:
    """
    Get a valid move from the human player.
    Accepts moves in UCI format (e.g., 'e2e4', 'e7e8q' for promotion).
    """
    while True:
        print("Your move (UCI format, e.g., 'e2e4' or 'type 'help' for legal moves, 'quit' to exit): ", end="")
        move_input = input().strip().lower()
        
        if move_input == 'quit':
            print("Thanks for playing!")
            exit(0)
        
        if move_input == 'help':
            print("\nLegal moves:")
            legal_moves = list(board.legal_moves)
            for i, move in enumerate(legal_moves):
                print(f"  {move.uci()}", end="")
                if (i + 1) % 8 == 0:
                    print()
            print("\n")
            continue
        
        try:
            move = chess.Move.from_uci(move_input)
            if move in board.legal_moves:
                return move
            else:
                print(f"❌ Illegal move: {move_input}. Try again.")
        except ValueError:
            print(f"❌ Invalid format: {move_input}. Use UCI notation (e.g., 'e2e4').")


def get_ai_move(game: ChessGame, board: chess.Board, mcts: MCTS, player: int, verbose: bool = True) -> chess.Move:
    """
    Get the AI's move using MCTS.
    
    Args:
        game: ChessGame instance
        board: Current board state
        mcts: MCTS instance
        player: Current player (1 for White, -1 for Black)
        verbose: Whether to print thinking information
    
    Returns:
        chess.Move object
    """
    if verbose:
        print("AI is thinking...")
    
    # Get canonical board for MCTS
    canonical_board = game.getCanonicalForm(board, player)
    
    # Get action probabilities from MCTS
    pi = mcts.getActionProb(canonical_board, temp=0)
    action = np.argmax(pi)
    
    # Decode action to move
    move = decode_move_index(board, action)
    
    if verbose:
        print(f"🤖 AI plays: {move.uci()} ({move.san(board)})")
    
    return move


def play_game(checkpoint_path: str = 'models/supervised_best.pth.tar', 
              num_mcts_sims: int = 50, 
              human_color: str = 'white'):
    """
    Main game loop for human vs AI chess.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        num_mcts_sims: Number of MCTS simulations per move
        human_color: 'white' or 'black'
    """
    # Initialize game components
    print("Initializing chess engine...")
    game = ChessGame()
    
    print(f"Loading model from {checkpoint_path}...")
    nnet = NNetWrapper(game)
    nnet.load_checkpoint('models/', checkpoint_path.split('/')[-1])
    
    print(f"Initializing MCTS with {num_mcts_sims} simulations per move...")
    args = type('Args', (), {'numMCTSSims': num_mcts_sims, 'cpuct': 1.0})()
    mcts = MCTS(game, nnet, args)
    
    # Initialize board
    board = game.getInitBoard()
    player = 1  # 1 = White, -1 = Black
    
    human_is_white = (human_color.lower() == 'white')
    
    print("\n" + "-" * 40)
    print("Chess Game: Human vs AI")
    print("-" * 40)
    print(f"You are playing as: {'White ♔' if human_is_white else 'Black ♚'}")
    print(f"AI is playing as: {'Black ♚' if human_is_white else 'White ♔'}")
    print("\nCommands:")
    print("  - Enter moves in UCI format (e.g., 'e2e4')")
    print("  - Type 'help' to see legal moves")
    print("  - Type 'quit' to exit")
    print("-" * 40)
    
    move_count = 0
    
    # Game loop
    while not board.is_game_over():
        display_board(board)
        
        # Determine whose turn it is
        is_human_turn = (board.turn == chess.WHITE and human_is_white) or \
                       (board.turn == chess.BLACK and not human_is_white)
        
        if is_human_turn:
            # Human's turn
            move = get_human_move(board)
        else:
            # AI's turn
            move = get_ai_move(game, board, mcts, player)
        
        # Make the move
        board.push(move)
        move_count += 1
        
        # Switch player
        player = -player
    
    # Game over
    display_board(board)
    print("\n" + "=" * 40)
    print("🏁 GAME OVER 🏁")
    print("=" * 40)
    
    # Determine result
    result = board.result()
    if result == "1-0":
        winner = "White wins!" if human_is_white else "AI wins!"
        print(f"Checkmate! {winner} ♔")
    elif result == "0-1":
        winner = "Black wins!" if not human_is_white else "AI wins!"
        print(f"Checkmate! {winner} ♚")
    else:
        print(f"Draw! Result: {result}")
        if board.is_stalemate():
            print("Reason: Stalemate")
        elif board.is_insufficient_material():
            print("Reason: Insufficient material")
        elif board.is_fifty_moves():
            print("Reason: Fifty-move rule")
        elif board.is_repetition():
            print("Reason: Threefold repetition")
    
    print(f"\nTotal moves: {move_count}")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(description="Play chess against the trained AI")
    parser.add_argument('--checkpoint', type=str, default='supervised_best.pth.tar',
                       help='Model checkpoint filename (default: supervised_best.pth.tar)')
    parser.add_argument('--mcts_sims', type=int, default=50,
                       help='Number of MCTS simulations per move (default: 50)')
    parser.add_argument('--color', type=str, default='white', choices=['white', 'black'],
                       help='Color to play as (default: white)')
    
    args = parser.parse_args()
    
    try:
        play_game(
            checkpoint_path=args.checkpoint,
            num_mcts_sims=args.mcts_sims,
            human_color=args.color
        )
    except KeyboardInterrupt:
        print("\n\nGame interrupted. Thanks for playing!")
    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        print("Make sure you have a trained model checkpoint in the models/ directory.")
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
