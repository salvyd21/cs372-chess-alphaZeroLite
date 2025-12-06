from core.Coach import Coach
from chess.ChessGame import ChessGame
from chess_nnet.NNetWrapper import NNetWrapper

def main():
    game = ChessGame()
    nnet = NNetWrapper(game)

    coach = Coach(game, nnet, args)
    coach.learn()

if __name__ == "__main__":
    main()