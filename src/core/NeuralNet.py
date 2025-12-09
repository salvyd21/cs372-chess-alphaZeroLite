class NeuralNet():
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below.
    """
    def __init__(self, game):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nnet = ChessResNet(game).to(self.device)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        
        # Training params
        self.lr = 0.001
        self.dropout = 0.3
        self.epochs = 10
        self.batch_size = 64
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=self.lr)

    def train(self, examples):
        """
        This function trains the neural network with examples obtained from
        self-play.
        
        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector,
                      v is its value.
        """
        self.nnet.train()
        
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            
            batch_count = int(len(examples) / self.batch_size)
            
            # Helper to generate batches
            for i in range(batch_count):
                sample_ids = np.random.randint(len(examples), size=self.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                
                # Convert to tensors
                boards = torch.FloatTensor(np.array(boards).astype(np.float64)).to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).to(self.device)

                # Predict
                out_pi, out_v = self.nnet(boards)

                # Loss calculation
                # Policy Loss: Cross Entropy (using log_softmax output + target probs)
                l_pi = -torch.sum(target_pis * out_pi) / target_pis.size()[0]
                
                # Value Loss: Mean Squared Error
                l_v = torch.sum((target_vs - out_v.view(-1)) ** 2) / target_vs.size()[0]
                
                total_loss = l_pi + l_v

                # Backprop
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
            print(f"Latest Loss: PI: {l_pi.item():.4f}, V: {l_v.item():.4f}")

    def predict(self, board):
        """
        Input:
            board: current board in its canonical form.
        Returns:
            pi: a policy vector for the current board - numpy array of length game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        # Prepare input
        board = torch.FloatTensor(board.astype(np.float64)).to(self.device)
        board = board.view(1, 12, self.board_x, self.board_y) # Add batch dim
        
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # Return as numpy
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        if not os.path.exists(folder):
            os.mkdir(folder)
            
        filepath = os.path.join(folder, filename)
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
            
        checkpoint = torch.load(filepath, map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
