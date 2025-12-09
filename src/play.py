# play.py (Simple example)
from chess.ChessGame import ChessGame
from chess.pytorch.NNet import NNetWrapper as NNet
from mcts.MCTS import MCTS
import numpy as np

game = ChessGame()
nnet = NNet(game)
# Load your supervised checkpoint!
nnet.load_checkpoint('models/', 'supervised_epoch_10.pth.tar')

# Initialize MCTS with the trained brain
args = {'numMCTSSims': 50, 'cpuct': 1.0}
mcts = MCTS(game, nnet, args)

board = game.getInitBoard()
player = 1

while True:
    print("Your Turn:")
    # ... logic to get your move ...
    
    # AI Turn
    # It thinks by running simulations using the supervised NNet
    pi = mcts.getActionProb(board, temp=0)
    action = np.argmax(pi)
    
    board, player = game.getNextState(board, player, action)
    print("AI played action:", action)