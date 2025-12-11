
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Ensure src is in python path
project_root = '/content/cs372-chess-alphaZeroLite'
sys.path.append(os.path.join(project_root, 'src'))

# Corrected imports based on file structure
from chess_engine.ChessGame import ChessGame
from chess_nnet.ChessNNet import ChessResNet

class ChessMoveDataset(Dataset):
    def __init__(self, filepath):
        print(f"Loading data from {filepath}...")
        try:
            data = np.load(filepath)
            self.X = data['X']
            self.y = data['y']
            print(f"Dataset loaded: {len(self.X)} samples.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.X = np.array([])
            self.y = np.array([])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        board = torch.from_numpy(self.X[idx]).float()
        move = torch.tensor(self.y[idx], dtype=torch.long)
        return board, move

def train_supervised():
    # Configuration
    args = {
        'batch_size': 256,
        'epochs': 8,             # Train for 8 more epochs
        'lr': 0.001,
        'num_channels': 512,
        'num_res_blocks': 20,
        'dropout': 0.3,
        'cuda': torch.cuda.is_available()
    }

    start_epoch = 2 # Resuming after epoch 2

    device = torch.device('cuda' if args['cuda'] else 'cpu')
    print(f"Training on: {device}")

    # --- 1. Setup Data ---
    data_path = os.path.join(project_root, 'data/processed/supervised_train.npz')
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return

    dataset = ChessMoveDataset(data_path)
    if len(dataset) == 0:
        print("Dataset empty. Exiting.")
        return

    loader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True, num_workers=0)

    # --- 2. Setup Model ---
    game = ChessGame()

    # Initialize model
    checkpoint_path = os.path.join(project_root, 'models/supervised_retrained_epoch_2.pth')

    # Helper class for unpickling legacy checkpoints (if needed)
    class Args:
        pass

    model_state = None
    loaded_args = None
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}. Loading...")
        try:
            try:
                 checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            except TypeError:
                 checkpoint = torch.load(checkpoint_path, map_location=device)

            if 'args' in checkpoint:
                loaded_args = checkpoint['args']
                # Map object to dict if necessary
                if not isinstance(loaded_args, dict):
                    if hasattr(loaded_args, '__dict__'):
                        loaded_args = vars(loaded_args)

                print(f"Updating model args from checkpoint: {loaded_args}")
                for k in ['num_channels', 'num_res_blocks', 'dropout']:
                    if k in loaded_args:
                        args[k] = loaded_args[k]

            model_state = checkpoint.get('state_dict', checkpoint)
            optimizer_state = checkpoint.get('optimizer', None)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return
    else:
        print(f"Checkpoint {checkpoint_path} not found! Cannot resume training.")
        return

    # Re-init model with potentially updated args
    model = ChessResNet(game, args).to(device)

    if model_state:
        try:
            model.load_state_dict(model_state)
            print("Successfully loaded model weights.")
        except Exception as e:
            print(f"Failed to load state dict (architecture mismatch?): {e}")
            return

    # --- 3. Optimizer & Loss ---
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    if 'optimizer_state' in locals() and optimizer_state:
        try:
            optimizer.load_state_dict(optimizer_state)
            print("Optimizer state loaded.")
        except Exception as e:
            print(f"Failed to load optimizer state: {e}")

    loss_fn = nn.NLLLoss()

    # --- 4. Training Loop ---
    model.train()

    for epoch in range(args['epochs']):
        actual_epoch = start_epoch + epoch + 1
        print(f"Starting Epoch {actual_epoch}...")
        
        total_loss = 0.0
        count = 0

        for batch_idx, (boards, moves) in enumerate(loader):
            boards = boards.to(device)
            moves = moves.to(device)

            optimizer.zero_grad()

            pi_pred, v_pred = model(boards)

            loss = loss_fn(pi_pred, moves)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {actual_epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(1, count)
        print(f"==> Epoch {actual_epoch} Finished. Avg Loss: {avg_loss:.4f}")

        # Save Checkpoint
        save_path = os.path.join(project_root, f"models/supervised_retrained_epoch_{actual_epoch}.pth")
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': args
        }, save_path)
        print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    train_supervised()
