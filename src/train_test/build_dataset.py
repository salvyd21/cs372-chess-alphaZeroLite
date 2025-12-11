import os
import sys
from pathlib import Path
import argparse
import io

# Add project root to sys.path to allow imports from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import chess
import chess.pgn
from tqdm import tqdm

from src.chess_engine.action_encoding import ACTION_SIZE, encode_move_index
from src.chess_engine.state_encoding import board_to_tensor

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]


# -----------------------------
# CANONICALIZATION HELPERS
# -----------------------------

def canonicalize_board_and_move(board: chess.Board, move: chess.Move):
    if board.turn == chess.WHITE:
        return board.copy(), move

    mirrored_board = board.copy().mirror()
    from_sq_m = chess.square_mirror(move.from_square)
    to_sq_m = chess.square_mirror(move.to_square)
    canonical_move = chess.Move(from_sq_m, to_sq_m, promotion=move.promotion)

    return mirrored_board, canonical_move


# -----------------------------
# PGN → EXAMPLES
# -----------------------------

def pgn_to_examples(
    pgn_path: Path,
    max_games: int | None = None,
    max_positions_per_game: int = 30,
    sample_every_n_moves: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    X = []
    y = []

    print(f"Reading PGN from: {pgn_path}")
    if str(pgn_path).endswith(".zst"):
        import zstandard as zstd
        fh = open(pgn_path, "rb")
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(fh)
        f = io.TextIOWrapper(stream_reader, encoding="utf-8")
    else:
        f = open(pgn_path, "r", encoding="utf-8")
    with f:
        game_count = 0

        while True:
            try:
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
                    if ply_idx % sample_every_n_moves != 0:
                        board.push(move)
                        continue

                    canon_board, canon_move = canonicalize_board_and_move(board, move)
                    positions.append(canon_board)
                    moves.append(canon_move)

                    board.push(move)

                if len(positions) > max_positions_per_game:
                    idxs = np.linspace(0, len(positions) - 1,
                                       max_positions_per_game, dtype=int)
                    positions = [positions[i] for i in idxs]
                    moves = [moves[i] for i in idxs]

                for board_pos, move in zip(positions, moves):
                    X.append(board_to_tensor(board_pos))
                    idx = encode_move_index(board_pos, move)
                    y.append(idx)

            except Exception as e:
                print(f"Error processing game {game_count}: {e}")
                continue

    X = np.stack(X) if X else np.empty((0, 12, 8, 8), dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"Total examples: {len(y)} (ACTION_SIZE = {ACTION_SIZE})")
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

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

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
    parser.add_argument("--pgn", type=str, required=True)
    parser.add_argument("--max_games", type=int, default=20000)
    parser.add_argument("--max_positions_per_game", type=int, default=30)
    parser.add_argument("--sample_every_n_moves", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="data/processed")
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
    np.savez_compressed(out_dir / "supervised_val.npz",   X=X_val,   y=y_val)
    np.savez_compressed(out_dir / "supervised_test.npz",  X=X_test,  y=y_test)

    print("Saved:")
    print(f"  train -> {out_dir / 'supervised_train.npz'} ({X_train.shape[0]} examples)")
    print(f"  val   -> {out_dir / 'supervised_val.npz'}   ({X_val.shape[0]} examples)")
    print(f"  test  -> {out_dir / 'supervised_test.npz'}  ({X_test.shape[0]} examples)")


if __name__ == "__main__":
    main()
