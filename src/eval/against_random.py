import argparse
import random
import chess

from chess.ChessGame import ChessGame
from chess_nnet.NNetWrapper import NNetWrapper
from core.MCTS import MCTS

# TODO: replace with real args/config
class DummyArgs:
    numMCTSSims = 50
    cpuct = 1.0

def play_one_game(game, nnet, args, verbose=False):
    """
    Model (MCTS+NN) as White vs random bot as Black.
    Returns:
        +1 if model wins
        -1 if model loses
        0 for draw
    """
    board = game.getInitBoard()
    player = 1  # model starts as 'current player' and we treat that as White

    mcts = MCTS(game, nnet, args)

    while True:
        if verbose:
            print(board)
            print("Turn:", "White" if board.turn == chess.WHITE else "Black")

        if player == 1:
            # Model move via MCTS
            canonical_board = game.getCanonicalForm(board, player=1)
            pi = mcts.getActionProb(canonical_board, temp=0)
            action = int(max(range(len(pi)), key=lambda a: pi[a]))
            board, player = game.getNextState(board, player, action)
        else:
            # Random move
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            board.push(move)
            player = -player

        # Check end condition
        if board.is_game_over():
            result = board.result()  # "1-0", "0-1", "1/2-1/2"
            if result == "1-0":
                return 1
            elif result == "0-1":
                return -1
            else:
                return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, required=True)
    args_cli = parser.parse_args()

    game = ChessGame()
    nnet = NNetWrapper(game)
    nnet.load_checkpoint(folder="models", filename=args_cli.checkpoint)

    args = DummyArgs()

    wins = draws = losses = 0
    for i in range(args_cli.games):
        print(f"Game {i+1}/{args_cli.games}")
        result = play_one_game(game, nnet, args, verbose=False)
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1

    print(f"Results vs random: {wins}W {losses}L {draws}D")

if __name__ == "__main__":
    main()
