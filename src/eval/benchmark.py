import argparse
import time
import sys
import os
import torch
import torch.optim as optim
from pathlib import Path
import chess

# Ensure src is in python path
project_root = "/content/cs372-chess-alphaZeroLite"
if project_root not in sys.path:
    sys.path.append(os.path.join(project_root, "src"))

from chess_engine.ChessGame import ChessGame
from chess_nnet.NNetWrapper import NNetWrapper
from chess_nnet.ChessNNet import ChessResNet
from core.MCTS import MCTS

class Args:
    pass

class DummyArgs:
    def __init__(self):
        self.numMCTSSims = 25
        self.cpuct = 1.0

def forward_pass(nnet, game, iters=100):
    board = game.getInitBoard()
    canonical = game.getCanonicalForm(board, 1)
    
    print("Running forward pass benchmark...")
    # Warm-up
    for _ in range(5): _ = nnet.predict(canonical)
    
    start = time.time()
    for _ in range(iters):
        _ = nnet.predict(canonical)
    end = time.time()
    
    print(f"Forward passes: {iters}")
    print(f"Total time: {end - start:.4f}s")
    print(f"Average per forward: {(end - start)/iters:.6f}s")

def benchmark_mcts(nnet, game, args, iters=10):
    print(f"Running MCTS benchmark ({iters} iterations)...")
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

    # Load Architecture from Checkpoint
    print("Loading checkpoint config...")
    nnet_args = None
    try:
        try:
            checkpoint = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(args_cli.checkpoint, map_location="cpu")
            
        training_args = checkpoint.get("args", {})
        
        class Config:
            def __init__(self, dictionary):
                for k, v in dictionary.items(): setattr(self, k, v)
            def __getattr__(self, item): return None
            
        if isinstance(training_args, dict): nnet_args = Config(training_args)
        else: nnet_args = training_args
        
        nc = getattr(nnet_args, "num_channels", "default")
        nrb = getattr(nnet_args, "num_res_blocks", "default")
        print(f"Checkpoint config: num_channels={nc} num_res_blocks={nrb}")
        
    except Exception as e:
        print(f"Failed to load config: {e}")

    game = ChessGame()
    nnet = NNetWrapper(game)

    if nnet_args:
        print("Updating NNet architecture...")
        if getattr(nnet_args, "num_channels", None): nnet.args["num_channels"] = nnet_args.num_channels
        if getattr(nnet_args, "num_res_blocks", None): nnet.args["num_res_blocks"] = nnet_args.num_res_blocks
        if getattr(nnet_args, "dropout", None): nnet.args["dropout"] = nnet_args.dropout
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nnet.nnet = ChessResNet(game, nnet.args).to(device)
        nnet.optimizer = optim.Adam(nnet.nnet.parameters(), lr=nnet.args["lr"])

    nnet.load_checkpoint(str(Path(args_cli.checkpoint).parent), str(Path(args_cli.checkpoint).name))

    device = getattr(nnet, "device", torch.device("cuda"))
    print("Using device:", device)

    forward_pass(nnet, game, iters=args_cli.forward_iters)
    
    args = DummyArgs()
    benchmark_mcts(nnet, game, args, iters=args_cli.mcts_iters)

if __name__ == "__main__":
    main()
