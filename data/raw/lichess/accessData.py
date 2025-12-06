from pathlib import Path
import chess.pgn

# Path to PGN file
pgn_path = Path("data/raw/lichess/lichess_db_standard_rated_2025-01.pgn")

pgn = open(pgn_path, encoding="utf-8")

game = chess.pgn.read_game(pgn)
while game is not None:
    # process game
    game = chess.pgn.read_game(pgn)
