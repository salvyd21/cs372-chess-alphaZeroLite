import numpy as np
import chess

PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Convert a python-chess Board into a (C, 8, 8) float32 tensor.
    C = 12: [white P,N,B,R,Q,K, black P,N,B,R,Q,K]
    Board is assumed to be from the current player's POV if you're using canonicalization.
    """
    planes = np.zeros((12, 8, 8), dtype=np.float32)

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        pt = piece.piece_type
        color = piece.color  # True=White, False=Black
        file = chess.square_file(sq)  # 0..7 a..h
        rank = chess.square_rank(sq)  # 0..7 1..8 (from White's POV)

        type_idx = PIECE_TYPES.index(pt)  # 0..5
        channel = type_idx if color == chess.WHITE else 6 + type_idx
        planes[channel, rank, file] = 1.0

    return planes
