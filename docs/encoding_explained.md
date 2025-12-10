# Chess Action Encoding

## Overview
Actions are encoded as a 73-plane action space over an 8×8 board = 4,672 total actions.

## Action Planes (73 total)
- **Planes 0-55 (56 planes):** Sliding moves in 8 directions × 7 distances
  - Directions: up, down, left, right, up-right, up-left, down-right, down-left
- **Planes 56-63 (8 planes):** Knight moves (L-shaped jumps)
- **Planes 64-72 (9 planes):** Underpromotions (3 directions × 3 piece types: Knight, Bishop, Rook)
  - Queen promotions use normal sliding planes

## Example
A move from e2 to e4 (up 2 squares) maps to:
- Direction: 0 (up)
- Distance: 2
- From square: (1, 4) in (rank, file)
- Action index: calculated via encoding function