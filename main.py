import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from src.chess.ChessGame import ChessGame
from src.chess_nnet.NNetWrapper import NNetWrapper
from src.chess.action_encoding import ACTION_SIZE

# Args / Config

class Args:
    # --- model architecture ---
    num_channels = 128
    num_resBlocks = 5
    policy_channels = 32
    value_channels = 32
    dropout = 0.1
    use_batchnorm = True

    # --- training ---
    batch_size = 256
    lr = 1e-3
    weight_decay = 1e-4
    epochs = 10
    policy_loss_weight = 1.0
    value_loss_weight = 1.0
    dropout = 0.3
    l2 = 1e-4

    # --- device ---
    use_gpu = True
    device = "cuda"      # or "cpu" if unavailable
    num_workers = 4

    # --- paths ---
    train_npz = "data/processed/supervised_train.npz"
    val_npz   = "data/processed/supervised_val.npz"
    test_npz  = "data/processed/supervised_test.npz"
    models_dir = "models"

# Dataset for supervised training on NPZ

class ChessMoveDataset(Dataset):
    """
    Expects an .npz with:
      X: (N, C, 8, 8) float32 board tensors
      y: (N,) int64 move indices in [0, ACTION_SIZE)
    """
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X = data["X"]  # (N, C, 8, 8)
        self.y = data["y"]  # (N,)
        assert self.X.shape[0] == self.y.shape[0], "X and y length mismatch"

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])      # float32
        y = int(self.y[idx])
        return x, y


# Supervised Training

def train_supervised(args: Args, checkpoint_name: str):
    """
    Train the Chess NN on the preprocessed PGN dataset (policy-only).
    - Uses CrossEntropyLoss on move indices.
    - Optionally later you can add a value head if you generate game outcomes.
    """
    device = torch.device(args.device if args.use_gpu and torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    game = ChessGame()
    nnet = NNetWrapper(game, args)
    nnet.device = device
    nnet.nnet.to(device)

    # optimizer (already created in NNetWrapper.__init__ typically,
    # but we can override lr/weight_decay if needed)
    optimizer = nnet.optimizer

    # Datasets
    train_ds = ChessMoveDataset(args.train_npz)
    val_ds   = ChessMoveDataset(args.val_npz)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Loss for policy: CrossEntropy over ACTION_SIZE
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        nnet.nnet.train()
        running_loss = 0.0
        total = 0

        for X, y in train_loader:
            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)

            optimizer.zero_grad()

            # forward: ChessNNet returns (policy_logits, value)
            policy_logits, _ = nnet.nnet(X)
            # policy_logits: (B, ACTION_SIZE)
            loss = criterion(policy_logits, y)

            loss.backward()
            optimizer.step()

            batch_size = X.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

        train_loss = running_loss / total

        nnet.nnet.eval()
        val_loss = 0.0
        val_total = 0
        correct_top1 = 0

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.long)

                policy_logits, _ = nnet.nnet(X)
                loss = criterion(policy_logits, y)

                batch_size = X.size(0)
                val_loss += loss.item() * batch_size
                val_total += batch_size

                preds = policy_logits.argmax(dim=-1)
                correct_top1 += (preds == y).sum().item()

        val_loss /= val_total
        val_top1 = correct_top1 / val_total

        print(f"[Epoch {epoch}/{args.epochs}] "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_top1={val_top1:.4f}")

        # Save best checkpoint
        os.makedirs(args.models_dir, exist_ok=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  -> New best model, saving to {checkpoint_name}")
            nnet.save_checkpoint(folder=args.models_dir, filename=checkpoint_name)


# Test-set evaluation

def eval_on_testset(args: Args, checkpoint_name: str):
    """
    Load a trained model and evaluate top-1 / top-3 accuracy on test set.
    """
    device = torch.device(args.device if args.use_gpu and torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    game = ChessGame()
    nnet = NNetWrapper(game, args)
    nnet.load_checkpoint(folder=args.models_dir, filename=checkpoint_name)
    nnet.nnet.to(device)
    nnet.nnet.eval()

    test_ds = ChessMoveDataset(args.test_npz)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    correct_top1 = 0
    correct_top3 = 0
    total = 0

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)

            logits, _ = nnet.nnet(X)
            probs = torch.softmax(logits, dim=-1)

            # top-1
            top1 = probs.argmax(dim=-1)
            correct_top1 += (top1 == y).sum().item()

            # top-3
            topk_vals, topk_idx = probs.topk(3, dim=-1)
            match_top3 = (topk_idx == y.unsqueeze(-1)).any(dim=-1)
            correct_top3 += match_top3.sum().item()

            total += X.size(0)

    print(f"Test examples: {total}")
    print(f"Top-1 accuracy: {correct_top1 / total:.4f}")
    print(f"Top-3 accuracy: {correct_top3 / total:.4f}")


# modes = train_supervised & eval_test
def main():
    parser = argparse.ArgumentParser(description="Chess AlphaZero-style main driver")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # train_supervised
    p_train = subparsers.add_parser("train", help="Train supervised policy on PGN-derived dataset")
    p_train.add_argument("--checkpoint", type=str, default="supervised_best.pth",
                         help="Checkpoint filename inside models/")

    # eval_testset
    p_eval = subparsers.add_parser("eval_test", help="Evaluate trained model on test set")
    p_eval.add_argument("--checkpoint", type=str, default="supervised_best.pth",
                        help="Checkpoint filename inside models/")

    args_cli = parser.parse_args()
    args = Args()   # you could also load from a config file

    if args_cli.mode == "train":
        train_supervised(args, checkpoint_name=args_cli.checkpoint)
    elif args_cli.mode == "eval_test":
        eval_on_testset(args, checkpoint_name=args_cli.checkpoint)
    else:
        raise ValueError(f"Unknown mode {args_cli.mode}")


if __name__ == "__main__":
    main()
