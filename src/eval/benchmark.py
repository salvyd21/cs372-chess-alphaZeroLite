import argparse
import time
import torch
import chess

from chess_engine.ChessGame import ChessGame
from chess_nnet.NNetWrapper import NNetWrapper
from core.MCTS import MCTS

class DummyArgs:
    numMCTSSims = 25
    cpuct = 1.0

def forward_pass(nnet, game, iters=100):
    board = game.getInitBoard()
    canonical = game.getCanonicalForm(board, 1)

    # Warm-up
    for _ in range(5):
        _ = nnet.predict(canonical)

    start = time.time()
    for _ in range(iters):
        _ = nnet.predict(canonical)
    end = time.time()

    print(f"Forward passes: {iters}")
    print(f"Total time: {end - start:.4f}s")
    print(f"Average per forward: {(end - start)/iters:.6f}s")

def benchmark_mcts(nnet, game, args, iters=10):
    mcts = MCTS(game, nnet, args)
    board = game.getInitBoard()
    canonical = game.getCanonicalForm(board, 1)

    start = time.time()
    for _ in range(iters):
        _ = mcts.getActionProb(canonical, temp=0)
    end = time.time()

    print(f"MCTS calls: {iters}")
    print(f"Total time: {end - start:.4f}s")
    print(f"Average per call: {(end - start)/iters:.6f}s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--forward_iters", type=int, default=100)
    parser.add_argument("--mcts_iters", type=int, default=10)
    args_cli = parser.parse_args()

    game = ChessGame()
    nnet = NNetWrapper(game)
    nnet.load_checkpoint("models", args_cli.checkpoint)

    device = getattr(nnet, "device", torch.device("cuda"))
    print("Using device:", device)

    forward_pass(nnet, game, iters=args_cli.forward_iters)

    args = DummyArgs()
    benchmark_mcts(nnet, game, args, iters=args_cli.mcts_iters)

if __name__ == "__main__":
    main()
