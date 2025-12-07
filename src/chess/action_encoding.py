import chess
import numpy as np
import torch
## Alpha-Zero style action encoding with the 8x8 dimension board and 73 action planes consisting of 
## "sliding" moves, knight moves, and promotion actions.
# 8 sliding directions: up/down/left/right + diagonals (from White POV)
SLIDING_DIRS = [
    ( 1,  0),  # 0: up
    (-1,  0),  # 1: down
    ( 0,  1),  # 2: right
    ( 0, -1),  # 3: left
    ( 1,  1),  # 4: up-right
    ( 1, -1),  # 5: up-left
    (-1,  1),  # 6: down-right
    (-1, -1),  # 7: down-left
]

# Knight moves (dr, df), White POV
KNIGHT_DIRS = [
    ( 2,  1),
    ( 1,  2),
    (-1,  2),
    (-2,  1),
    (-2, -1),
    (-1, -2),
    ( 1, -2),
    ( 2, -1),
]

# Underpromotion directions from White POV:
# forward, capture-left, capture-right
PROMO_DIRS = [
    (1, 0),   # forward
    (1, -1),  # capture-left
    (1, 1),   # capture-right
]

# Underpromotion piece types (Queen promotion uses sliding planes)
PROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

NUM_SLIDING   = 8 * 7        # 56
NUM_KNIGHT    = 8            # 8
NUM_UNDERPROMO = 3 * 3       # 9
ACTION_PLANES = NUM_SLIDING + NUM_KNIGHT + NUM_UNDERPROMO  # 73
ACTION_SIZE   = 8 * 8 * ACTION_PLANES                      # 4672

def to_canonical(board: chess.Board, move: chess.Move):
    """
    Return (canonical_board, canonical_move, is_white_to_move)
    such that canonical_board.turn is always White.
    """
    if board.turn == chess.WHITE:
        return board, move, True
    else:
        # Mirror board and move for Black
        cb = board.mirror()
        cf = chess.square_mirror(move.from_square)
        ct = chess.square_mirror(move.to_square)
        cpromo = move.promotion
        cmove = chess.Move(cf, ct, promotion=cpromo)
        return cb, cmove, False


def from_canonical(board: chess.Board, move: chess.Move, was_white: bool):
    """
    Convert a canonical move (white POV) back to original side-to-move board.
    """
    if was_white:
        return move
    else:
        # Mirror back for Black
        of = chess.square_mirror(move.from_square)
        ot = chess.square_mirror(move.to_square)
        return chess.Move(of, ot, promotion=move.promotion)


def square_to_rf(square: int):
    """Return (rank, file) ∈ [0..7], white's POV."""
    file = chess.square_file(square)   # 0..7 (a..h)
    rank = chess.square_rank(square)   # 0..7 (1..8 from white)
    return rank, file

def encode_move_canonical(board: chess.Board, move: chess.Move):
    """
    Encode a move on a board where it's White to move (canonical).
    Returns (from_rank, from_file, plane).
    """
    from_sq = move.from_square
    to_sq   = move.to_square
    fr, ff = square_to_rf(from_sq)
    tr, tf = square_to_rf(to_sq)

    dr = tr - fr
    df = tf - ff

    # 1) Underpromotion (to N/B/R) -> planes 64..72
    if move.promotion in PROMO_PIECES:
        # Promotions from White POV: pawn must move one step forward to last rank
        # dr should be +1, and df ∈ {0, -1, +1}
        for d_idx, (pr, pf) in enumerate(PROMO_DIRS):
            if (dr, df) == (pr, pf):
                promo_piece_idx = PROMO_PIECES.index(move.promotion)
                promo_plane_idx = d_idx * len(PROMO_PIECES) + promo_piece_idx  # 0..8
                plane = NUM_SLIDING + NUM_KNIGHT + promo_plane_idx            # 64..72
                return fr, ff, plane

        raise ValueError(f"Underpromotion move {move} has unexpected delta {(dr, df)}")

    # 2) Knight moves -> planes 56..63
    for k_idx, (kr, kf) in enumerate(KNIGHT_DIRS):
        if (dr, df) == (kr, kf):
            plane = NUM_SLIDING + k_idx  # 56..63
            return fr, ff, plane

    # 3) Sliding moves (including king/queen/rook/bishop/pawn pushes/castling)
    if dr == 0 and df == 0:
        raise ValueError("Zero-length move?")

    # Normalize direction
    distance = max(abs(dr), abs(df))
    dr_unit = dr // distance if distance != 0 else 0
    df_unit = df // distance if distance != 0 else 0

    if (dr_unit, df_unit) not in SLIDING_DIRS:
        raise ValueError(f"Move {move} does not align with sliding directions or knight/underpromo")

    dir_idx = SLIDING_DIRS.index((dr_unit, df_unit))
    if not (1 <= distance <= 7):
        raise ValueError(f"Sliding distance {distance} out of range for move {move}")

    dist_idx = distance - 1  # 0..6
    plane = dir_idx * 7 + dist_idx  # 0..55

    return fr, ff, plane


def encode_move_index(board: chess.Board, move: chess.Move) -> int:
    """
    Full AlphaZero-style encoding:
    (board, move) -> flat index ∈ [0, ACTION_SIZE-1] = [0, 4671].
    Handles canonicalization for Black.
    """
    canon_board, canon_move, was_white = to_canonical(board, move)
    fr, ff, plane = encode_move_canonical(canon_board, canon_move)
    from_flat = fr * 8 + ff
    index = from_flat * ACTION_PLANES + plane
    return index

def decode_index_canonical(board: chess.Board, index: int) -> chess.Move:
    """
    Decode a flat index -> chess.Move on a canonical board (White to move).
    Does NOT check legality; that should be handled by masks / later checks.
    """
    if not (0 <= index < ACTION_SIZE):
        raise ValueError(f"Index {index} out of range 0..{ACTION_SIZE-1}")

    from_flat = index // ACTION_PLANES
    plane = index % ACTION_PLANES

    fr = from_flat // 8
    ff = from_flat % 8

    # Sliding moves
    if plane < NUM_SLIDING:
        dir_idx = plane // 7
        dist_idx = plane % 7
        distance = dist_idx + 1  # 1..7
        dr_unit, df_unit = SLIDING_DIRS[dir_idx]
        dr = dr_unit * distance
        df = df_unit * distance
        promotion = None

    # Knight moves
    elif plane < NUM_SLIDING + NUM_KNIGHT:
        k_idx = plane - NUM_SLIDING
        dr, df = KNIGHT_DIRS[k_idx]
        promotion = None

    # Underpromotions
    else:
        u_idx = plane - NUM_SLIDING - NUM_KNIGHT  # 0..8
        d_idx = u_idx // len(PROMO_PIECES)        # 0..2
        promo_idx = u_idx % len(PROMO_PIECES)     # 0..2

        dr, df = PROMO_DIRS[d_idx]
        promotion = PROMO_PIECES[promo_idx]

    tr = fr + dr
    tf = ff + df

    # If move goes off-board, it's invalid in practice
    if not (0 <= tr < 8 and 0 <= tf < 8):
        # You can raise or return a dummy move; I prefer raising.
        raise ValueError(f"Decoded move goes off-board: from ({fr},{ff}) to ({tr},{tf})")

    from_sq = chess.square(ff, fr)
    to_sq   = chess.square(tf, tr)

    move = chess.Move(from_sq, to_sq, promotion=promotion)
    return move

def decode_move_index(board: chess.Board, index: int) -> chess.Move:
    """
    Full AlphaZero-style decoding:
    (board, index) -> chess.Move for the original board.turn side.
    """
    # Make canonical copy (for White POV)
    if board.turn == chess.WHITE:
        canon_board = board
        was_white = True
    else:
        canon_board = board.mirror()
        was_white = False

    canon_move = decode_index_canonical(canon_board, index)
    move = from_canonical(board, canon_move, was_white)
    return move

def legal_move_mask(board: chess.Board, device="cpu"):
    mask = torch.zeros(ACTION_SIZE, dtype=torch.bool, device=device)
    for move in board.legal_moves:
        try:
            idx = encode_move_index(board, move)
            mask[idx] = True
        except ValueError:
            # Some weird moves could fail encoding if we messed up; ideally none do.
            pass
    return mask

# Example usage:
logits = policy_head_output  # shape (4672,)
mask = legal_move_mask(board, logits.device)
logits = logits.clone()
logits[~mask] = -1e9
probs = torch.softmax(logits, dim=-1)
idx = torch.argmax(probs).item()
chosen_move = decode_move_index(board, idx)
