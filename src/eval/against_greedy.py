import argparse
import sys
import os
from pathlib import Path
import numpy as np
import chess
import torch
import torch.optim as optim

# Ensure src is in python path
project_root = "/content/cs372-chess-alphaZeroLite"
if project_root not in sys.path:
    sys.path.append(os.path.join(project_root, "src"))

from chess_engine.ChessGame import ChessGame
from chess_nnet.NNetWrapper import NNetWrapper
from chess_nnet.ChessNNet import ChessResNet
from core.MCTS import MCTS

# Helper class for unpickling legacy checkpoints
class Args:
    pass

class DummyArgs:
    def __init__(self):
        self.numMCTSSims = 50
        self.cpuct = 1.0

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
}

def greedy_move(board: chess.Board) -> chess.Move:
    best_move = None
    best_score = -999
    for move in board.legal_moves:
        score = 0
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                score += PIECE_VALUES.get(captured_piece.piece_type, 0)
        if score > best_score:
            best_score = score
            best_move = move
    if best_move is None:
        if board.legal_moves.count() > 0:
            best_move = list(board.legal_moves)[0]
        else:
            return None
    return best_move

def play_one_game(game, nnet, args, verbose=False):
    board = game.getInitBoard()
    player = 1
    mcts = MCTS(game, nnet, args)
    while True:
        if verbose: print(board)
        if player == 1:
            try:
                canonical_board = game.getCanonicalForm(board, 1)
                pi = mcts.getActionProb(canonical_board, temp=0)
                action = int(max(range(len(pi)), key=lambda a: pi[a]))
                board, player = game.getNextState(board, player, action)
            except Exception as e:
                print(f"Error during MCTS/Move selection: {e}")
                return -1
        else:
            move = greedy_move(board)
            if move is None: break
            board.push(move)
            player = -player
        if board.is_game_over():
            result = board.result()
            if result == "1-0": return 1
            elif result == "0-1": return -1
            else: return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, required=True)
    args_cli = parser.parse_args()

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
    args = DummyArgs()
    wins = draws = losses = 0
    for i in range(args_cli.games):
        result = play_one_game(game, nnet, args)
        if result == 1: wins += 1
        elif result == -1: losses += 1
        else: draws += 1
        print(f"Game {i+1}/{args_cli.games} Result: {result} (Cumulative: {wins}W {losses}L {draws}D)")
    print(f"Final Results vs Greedy: {wins}W {losses}L {draws}D")

if __name__ == "__main__":
    main()
