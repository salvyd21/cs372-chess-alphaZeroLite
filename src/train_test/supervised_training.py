sys.path.append(os.getcwd())

from chess_engine.pytorch.ChessNNet import ChessResNet
# Import your game class to get dimensions
from chess_engine.ChessGame import ChessGame 

class ChessMoveDataset(Dataset):
    def __init__(self, filepath):
        print(f"Loading data from {filepath}...")
        data = np.load(filepath)
        
        # X: (N, 12, 8, 8) - Board states
        # y: (N,)          - Move indices (integers)
        self.X = data['X']
        self.y = data['y']
        
        print(f"Dataset loaded: {len(self.X)} samples.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert numpy arrays to torch tensors
        # Board must be Float (for the CNN), Move must be Long (for the Loss lookup)
        board = torch.from_numpy(self.X[idx]).float()
        move = torch.tensor(self.y[idx], dtype=torch.long)
        
        return board, move

def train_supervised():
    # --- Configuration ---
    args = {
        'batch_size': 64,
        'epochs': 10,
        'lr': 0.001,
        'num_channels': 256,
        'num_res_blocks': 5,
        'cuda': torch.cuda.is_available()
    }
    
    device = torch.device('cuda' if args['cuda'] else 'cpu')
    print(f"Training on: {device}")

    # --- 1. Setup Data ---
    dataset = ChessMoveDataset('data/processed/supervised_train.npz')
    loader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True, num_workers=4)

    # --- 2. Setup Model ---
    game = ChessGame()
    model = ChessResNet(game, args).to(device)

    # --- 3. Optimizer & Loss ---
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    
    # We use NLLLoss because the model output (pi) already has LogSoftmax applied.
    loss_fn = nn.NLLLoss()

    # --- 4. Training Loop ---
    model.train()
    
    for epoch in range(args['epochs']):
        total_loss = 0.0
        
        # Iterate through the DataLoader
        for batch_idx, (boards, moves) in enumerate(loader):
            # Move data to GPU/CPU
            boards = boards.to(device)
            moves = moves.to(device)

            # Zero Gradients
            optimizer.zero_grad()

            # Forward Pass
            # The model returns (pi, v). For supervised moves, we only care about pi.
            pi_pred, v_pred = model(boards)

            # Calculate Loss
            # Compare predicted policy (pi_pred) vs actual move index (moves)
            loss = loss_fn(pi_pred, moves)

            # Backward Pass
            loss.backward()
            optimizer.step()

            # Tracking
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        # End of Epoch Stats
        avg_loss = total_loss / len(loader)
        print(f"==> Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}")

        # Save Checkpoint
        save_path = f"models/supervised_epoch_{epoch+1}.pth.tar"
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': args
        }, save_path)
        print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    train_supervised()
