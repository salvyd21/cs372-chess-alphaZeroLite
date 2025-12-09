from core.NeuralNet import NeuralNet
from chess_engine.state_encoding import board_to_tensor
from .ChessNNet import ChessResNet
import torch
import tqdm
import torch.optim as optim
import numpy as np
import os
import chess


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        # Configuration arguments (Adjust these hyperparameters as needed)
        self.args = {
            'lr': 0.001,
            'dropout': 0.3,
            'l2': 0.0001,
            'epochs': 10,
            'batch_size': 64,
            'cuda': torch.cuda.is_available(),
            'num_channels': 128,
            'num_res_blocks': 5
        }
        
        self.device = torch.device('cuda' if self.args['cuda'] else 'cpu')
        
        # Initialize the architecture with the args
        self.nnet = ChessResNet(game, self.args).to(self.device)
        
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=self.args['lr'])

    def train(self, examples):
        """
        examples: list of (canonicalBoard, pi, v)
        """
        optimizer = self.optimizer
        
        for epoch in range(self.args['epochs']):
            print(f'Epoch {epoch + 1}/{self.args["epochs"]}')
            self.nnet.train()
            
            batch_count = int(len(examples) / self.args['batch_size'])
            
            # Use tqdm for a progress bar
            t = tqdm.range(batch_count), desc='Training Net'
            for _ in t:
                # 1. Sample a batch
                sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                
                # 2. Convert boards to tensors
                # We assume training examples might already be tensors (from supervised loader)
                # or raw boards. If raw, we convert them.
                if isinstance(boards[0], np.ndarray) and len(boards[0].shape) == 3:
                     # Already encoded (12, 8, 8)
                    board_tensors = torch.FloatTensor(np.array(boards).astype(np.float64))
                else:
                    # Need conversion (if passed raw board objects)
                    board_tensors = torch.FloatTensor(np.array([board_to_tensor(b) for b in boards]).astype(np.float64))

                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # Move to GPU
                if self.args['cuda']:
                    board_tensors = board_tensors.contiguous().to(self.device)
                    target_pis = target_pis.contiguous().to(self.device)
                    target_vs = target_vs.contiguous().to(self.device)

                # 3. Compute Output & Loss
                out_pi, out_v = self.nnet(board_tensors)
                
                l_pi = -torch.sum(target_pis * out_pi) / target_pis.size()[0]
                l_v = torch.sum((target_vs - out_v.view(-1)) ** 2) / target_vs.size()[0]
                total_loss = l_pi + l_v

                # 4. Backprop
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                t.set_postfix(Loss_PI=l_pi.item(), Loss_V=l_v.item())

    def predict(self, canonicalBoard):
        """
        board: canonical board (chess.Board OR numpy array (12,8,8))
        Returns: pi (policy vector), v (value)
        """
        # 1. Prepare input — handle both cases chess.Board or numpy arrays
        if isinstance(canonicalBoard, np.ndarray):
            # Already encoded as (12, 8, 8)
            board_tensor = canonicalBoard.astype(np.float32)
        elif isinstance(canonicalBoard, chess.Board):
            # Raw board object — encode it
            board_tensor = board_to_tensor(canonicalBoard).astype(np.float32)
        else:
            raise TypeError(f"Expected chess.Board or numpy array, got {type(canonicalBoard)}")
    
        # 2. Convert to Torch and add batch dimension
        board_tensor = torch.FloatTensor(board_tensor).unsqueeze(0).to(self.device)
        
        # 3. Inference
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board_tensor)

        # 4. Return results
        return pi.cpu().data.numpy()[0], v.cpu().data.numpy()[0]

    def save_checkpoint(self, folder='models', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
            
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'args': self.args,
        }, filepath)

    def load_checkpoint(self, folder='models', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}")
            
        map_location = None if self.args['cuda'] else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        
        self.nnet.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
