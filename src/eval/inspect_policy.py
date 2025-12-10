import argparse
import numpy as np
import chess

from chess_engine.ChessGame import ChessGame
from chess_nnet.NNetWrapper import NNetWrapper
from chess_engine.action_encoding import encode_move_index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fen", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--topk", type=int, default=5)
    args_cli = parser.parse_args()

    game = ChessGame()
    nnet = NNetWrapper(game)
    nnet.load_checkpoint("models", args_cli.checkpoint)

    board = chess.Board(args_cli.fen)
    canonical = game.getCanonicalForm(board, player=1)

    # NN prediction without MCTS
    policy, value = nnet.predict(canonical)  # policy shape: (ACTION_SIZE,)

    print("Value prediction (from side to move POV):", float(value))

    # restrict to legal moves
    mask = game.getValidMoves(canonical, player=1)
    policy = np.array(policy)
    policy = policy * mask
    if policy.sum() > 0:
        policy = policy / policy.sum()

    # get k best moves
    legal_moves = list(canonical.legal_moves)
    scored_moves = []
    for move in legal_moves:
        idx = encode_move_index(canonical, move)
        scored_moves.append((move, policy[idx]))

    scored_moves.sort(key=lambda x: x[1], reverse=True)
    print(f"Top {args_cli.topk} moves by NN policy:")
    for move, p in scored_moves[:args_cli.topk]:
        print(f"  {canonical.san(move):8s}  prob={p:.4f}")

if __name__ == "__main__":
    main()

