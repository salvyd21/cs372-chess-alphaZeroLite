import chess
import numpy as np
import torch

def board_to_tensor(board: chess.Board) -> np.ndarray:
    # want to return (C,8,8) float32
    ...