import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from chess_engine.ChessGame import ChessGame
from chess_nnet.NNetWrapper import NNetWrapper
from chess_engine.action_encoding import ACTION_SIZE

class ChessTestDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = data["X"]  # (N, C, 8, 8)
        self.y = data["y"]  # (N,)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])  # float32
        y = int(self.y[idx])
        return x, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="data/processed/supervised_test.npz")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    args_cli = parser.parse_args()

    game = ChessGame()
    nnet = NNetWrapper(game)
    nnet.load_checkpoint("models", args_cli.checkpoint)
    nnet.nnet.eval()

    device = getattr(nnet, "device", torch.device("cpu"))

    dataset = ChessTestDataset(args_cli.npz)
    loader = DataLoader(dataset, batch_size=args_cli.batch_size, shuffle=False)

    correct_top1 = 0
    correct_top3 = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            # assume nnet.nnet.forward(X) returns (logits, value)
            logits, _ = nnet.nnet(X)
            # logits shape: (batch, ACTION_SIZE)
            probs = torch.softmax(logits, dim=-1)

            top1 = probs.argmax(dim=-1)
            correct_top1 += (top1 == y).sum().item()

            topk_vals, topk_idx = probs.topk(3, dim=-1)
            match_top3 = (topk_idx == y.unsqueeze(-1)).any(dim=-1)
            correct_top3 += match_top3.sum().item()

            total += X.size(0)

    print(f"Test examples: {total}")
    print(f"Top-1 accuracy: {correct_top1 / total:.4f}")
    print(f"Top-3 accuracy: {correct_top3 / total:.4f}")

if __name__ == "__main__":
    main()
