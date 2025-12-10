# Data Processing Pipeline

## Source Data
- **Dataset:** Lichess standard rated games (January 2025)
- **File:** `lichess_db_standard_rated_2025-01.pgn.zst`
- **Size:** ~30 GB compressed
- **Location:** `data/raw/lichess/`

## Processing Steps
1. Decompress `.pgn.zst` → `.pgn`
2. Parse PGN with `python-chess`
3. For each position-move pair:
   - Encode board state → (12, 8, 8) tensor
   - Encode move → integer in [0, 4672)
4. Save as compressed `.npz` files

## Output Format
```python
supervised_train.npz:
  X: (N, 12, 8, 8) float32  # Board states
  y: (N,) int64             # Move indices