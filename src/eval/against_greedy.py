import argparse
import chess

from chess_engine.ChessGame import ChessGame
from chess_nnet.NNetWrapper import NNetWrapper
from core.MCTS import MCTS

class DummyArgs:
    numMCTSSims = 50
    cpuct = 1.0

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

def greedy_move(board: chess.Board) -> chess.Move:
    """
    Choose legal move that maximizes immediate material gain.
    """
    best_move = None
    best_score = -999

    for move in board.legal_moves:
        score = 0
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                score += PIECE_VALUES.get(captured_piece.piece_type, 0)
        # TODO: optionally add positional heuristics
        if score > best_score:
            best_score = score
            best_move = move

    # fallback if nothing special
    if best_move is None:
        best_move = next(iter(board.legal_moves))
    return best_move

def play_one_game(game, nnet, args, verbose=False):
    """
    Model (MCTS+NN) as White vs greedy bot as Black.
    """
    board = game.getInitBoard()
    player = 1
    mcts = MCTS(game, nnet, args)

    while True:
        if verbose:
            print(board)

        if player == 1:
            canonical_board = game.getCanonicalForm(board, 1)
            pi = mcts.getActionProb(canonical_board, temp=0)
            action = int(max(range(len(pi)), key=lambda a: pi[a]))
            board, player = game.getNextState(board, player, action)
        else:
            move = greedy_move(board)
            board.push(move)
            player = -player

        if board.is_game_over():
            result = board.result()
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
    nnet.load_checkpoint("models", args_cli.checkpoint)

    args = DummyArgs()

    wins = draws = losses = 0
    for i in range(args_cli.games):
        print(f"Game {i+1}/{args_cli.games}")
        result = play_one_game(game, nnet, args)
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1

    print(f"Results vs greedy: {wins}W {losses}L {draws}D")

if __name__ == "__main__":
    main()
