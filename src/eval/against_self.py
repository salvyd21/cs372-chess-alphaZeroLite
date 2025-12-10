import argparse
import chess

from chess_engine.ChessGame import ChessGame
from chess_nnet.NNetWrapper import NNetWrapper
from core.MCTS import MCTS

class DummyArgs:
    numMCTSSims = 50
    cpuct = 1.0

def play_one_game(game, nnet1, nnet2, args, verbose=False):
    """
    Play one game between model A (White) and model B (Black).
    Returns:
        +1 if model A wins
        -1 if model B wins
        0 for draw
    """
    board = game.getInitBoard()
    player = 1

    mcts1 = MCTS(game, nnet1, args)
    mcts2 = MCTS(game, nnet2, args)

    while True:
        if verbose:
            print(board)

        current_nnet = nnet1 if player == 1 else nnet2
        current_mcts = mcts1 if player == 1 else mcts2

        canonical_board = game.getCanonicalForm(board, player)
        pi = current_mcts.getActionProb(canonical_board, temp=0)
        action = int(max(range(len(pi)), key=lambda a: pi[a]))
        board, player = game.getNextState(board, player, action)

        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return 1 # Model A (White) wins
            elif result == "0-1":
                return -1 # Model B (Black) wins
            else:
                return 0 # Draw

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelA", required=True)
    parser.add_argument("--modelB", required=True)
    parser.add_argument("--games", type=int, default=10)
    args_cli = parser.parse_args()

    game = ChessGame()

    nnetA = NNetWrapper(game)
    nnetB = NNetWrapper(game)
    nnetA.load_checkpoint("models", args_cli.modelA)
    nnetB.load_checkpoint("models", args_cli.modelB)

    args = DummyArgs()

    a_wins = b_wins = draws = 0
    for i in range(args_cli.games):
        print(f"Game {i+1}/{args_cli.games}")
        # A is always white for simplicity
        result = play_one_game(game, nnetA, nnetB, args)
        if result == 1:
            a_wins += 1
        elif result == -1:
            b_wins += 1
        else:
            draws += 1

    print(f"Model A vs Model B over {args_cli.games} games:")
    print(f"A wins: {a_wins}, B wins: {b_wins}, Draws: {draws}")

if __name__ == "__main__":
    main()

