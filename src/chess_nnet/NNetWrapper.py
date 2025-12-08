from core.NeuralNet import NeuralNet
from chess.state_encoding import board_to_tensor
from .ChessNNet import ChessNNet

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = ChessNNet(game, args)
        self.device = ...

    def train(self, examples):
        # examples = list of (canonicalBoard, pi, z)
        # or for supervised: (board_tensor, move_index)
        ...

    def predict(self, canonicalBoard):
        # board -> tensor -> nnet -> (policy, value)
        ...

    def save_checkpoint(...):
        ...

    def load_checkpoint(...):
        ...
