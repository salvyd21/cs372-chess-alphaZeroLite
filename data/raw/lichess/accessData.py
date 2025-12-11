
import requests
import zstandard as zstd
import chess.pgn
import io
import sys
import os

# Configuration
URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2014-01.pgn.zst"
OUTPUT_FILE = "/content/cs372-chess-alphaZeroLite/data/raw/lichess/games_200k.pgn"
TARGET_GAMES = 200000

def download_and_process():
    print(f"Starting download from {URL}...")
    print(f"Target: {TARGET_GAMES} games.")

    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Stream the response
    try:
        response = requests.get(URL, stream=True)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to connect to Lichess database: {e}")
        return

    dctx = zstd.ZstdDecompressor()

    # Use a stream reader for the zstd compressed data
    with dctx.stream_reader(response.raw) as reader:
        # Wrap in text mode for chess.pgn
        text_stream = io.TextIOWrapper(reader, encoding='utf-8')

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
            exporter = chess.pgn.FileExporter(out_f)
            count = 0
            errors = 0
            
            while count < TARGET_GAMES:
                try:
                    # Read game headers/moves
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break
                    
                    # Attempt to write the game to the output file
                    try:
                        game.accept(exporter)
                        count += 1
                        if count % 10000 == 0:
                            print(f"Saved {count} games... (Errors: {errors})")
                    except Exception as e:
                        # Skip this game if export fails
                        errors += 1
                        continue

                except Exception as e:
                    print(f"Error parsing game stream at index {count}: {e}")
                    errors += 1
                    continue

    print(f"Finished! Saved {count} games to {OUTPUT_FILE}. Total skipped errors: {errors}")

if __name__ == "__main__":
    download_and_process()
