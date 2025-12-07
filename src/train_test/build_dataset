import os
from pathlib import Path
import argparse

import numpy as np
import chess
import chess.pgn
from tqdm import tqdm


# -----------------------------
# CONFIG
# -----------------------------

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]

# (from_square, to_square, promotion) encoding
PROMO_MAP = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
NUM_PROMOS = len(PROMO_MAP)  # 5
ACTION_SIZE = 64 * 64 * NUM_PROMOS  # 20480


# -----------------------------
# ENCODING FUNCTIONS
# -----------------------------

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Encode board as (12, 8, 8) float32.
    Channel order:
    [WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK]
    """
    planes = np.zeros((12, 8, 8), dtype=np.float32)

    for color_idx, color in enumerate([chess.WHITE, chess.BLACK]):
        for pt_idx, piece_type in enumerate(PIECE_TYPES):
            channel = color_idx * 6 + pt_idx
            for square in board.pieces(piece_type, color):
                rank = 7 - chess.square_rank(square)  # rank 0 at top
                file = chess.square_file(square)      # file 0 = 'a'
                planes[channel, rank, file] = 1.0

    return planes


def move_to_index(move: chess.Move) -> int:
    from_sq = move.from_square
    to_sq = move.to_square
    promo_id = PROMO_MAP.get(move.promotion, 0)
    return from_sq * 64 * NUM_PROMOS + to_sq * NUM_PROMOS + promo_id


# -----------------------------
# PGN → EXAMPLES
# -----------------------------

def pgn_to_examples(
    pgn_path: Path,
    max_games: int | None = None,
    max_positions_per_game: int = 30,
    sample_every_n_moves: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a PGN file into (X, y):

    X: (N, 12, 8, 8) float32
    y: (N,) int64 action indices
    """
    X = []
    y = []

    print(f"Reading PGN from: {pgn_path}")
    with open(pgn_path, "r", encoding="utf-8") as f:
        game_count = 0

        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break  # EOF

            game_count += 1
            if max_games is not None and game_count > max_games:
                break

            if game_count % 1000 == 0:
                print(f"Processed {game_count} games...")

            board = game.board()
            positions = []
            moves = []

            for ply_idx, move in enumerate(game.mainline_moves()):
                # Optional: subsample moves (e.g., every 2nd or 3rd)
                if ply_idx % sample_every_n_moves != 0:
                    board.push(move)
                    continue

                positions.append(board.copy())
                moves.append(move)
                board.push(move)

            # Limit positions per game to control dataset size
            if len(positions) > max_positions_per_game:
                idxs = np.linspace(0, len(positions) - 1,
                                   max_positions_per_game, dtype=int)
                positions = [positions[i] for i in idxs]
                moves = [moves[i] for i in idxs]

            for board_pos, move in zip(positions, moves):
                X.append(board_to_tensor(board_pos))
                y.append(move_to_index(move))

    X = np.stack(X) if X else np.empty((0, 12, 8, 8), dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"Total examples: {len(y)}")
    return X, y


# -----------------------------
# TRAIN/VAL/TEST SPLIT
# -----------------------------

def train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, seed=42):
    assert X.shape[0] == y.shape[0]
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end] # 60% training
    val_idx = indices[train_end:val_end] # 20% validation
    test_idx = indices[val_end:] # 20% testing

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
    )


# -----------------------------
# MAIN
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pgn",
        type=str,
        required=True,
        help="Path to PGN file (e.g. data/raw/lichess/lichess_db.pgn)",
    )
    parser.add_argument(
        "--max_games",
        type=int,
        default=20000,
        help="Max number of games to parse (None = all)",
    )
    parser.add_argument(
        "--max_positions_per_game",
        type=int,
        default=30,
        help="Max positions to extract per game",
    )
    parser.add_argument(
        "--sample_every_n_moves",
        type=int,
        default=1,
        help="Take every Nth move from each game (1 = all moves)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed",
        help="Directory to save npz files",
    )
    args = parser.parse_args()

    pgn_path = Path(args.pgn)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = pgn_to_examples(
        pgn_path,
        max_games=args.max_games if args.max_games > 0 else None,
        max_positions_per_game=args.max_positions_per_game,
        sample_every_n_moves=args.sample_every_n_moves,
    )

    if X.shape[0] == 0:
        print("No examples generated. Check PGN path and settings.")
        return

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

    np.savez_compressed(out_dir / "supervised_train.npz", X=X_train, y=y_train)
    np.savez_compressed(out_dir / "supervised_val.npz", X=X_val, y=y_val)
    np.savez_compressed(out_dir / "supervised_test.npz", X=X_test, y=y_test)

    print("Saved:")
    print(f"  train -> {out_dir / 'supervised_train.npz'} ({X_train.shape[0]} examples)")
    print(f"  val   -> {out_dir / 'supervised_val.npz'}   ({X_val.shape[0]} examples)")
    print(f"  test  -> {out_dir / 'supervised_test.npz'}  ({X_test.shape[0]} examples)")


if __name__ == "__main__":
    main()
