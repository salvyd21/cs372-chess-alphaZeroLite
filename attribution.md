# Attribution

## Data Sources
- **Lichess Game Database**: [https://database.lichess.org/](https://database.lichess.org/)
  - File: `lichess_db_standard_rated_2025-01.pgn.zst`
  - License: CC0 1.0 Universal (Public Domain)
  - Used for supervised training/validating/testing of chess move/action prediction
  - Processed into using 50k games because the original db size of >90 million games is a little too ambitious
  - Randomly picked which games to process into usable data for the model.

## Libraries and Frameworks
- **PyTorch** (torch): [https://pytorch.org/](https://pytorch.org/)
  - License: BSD-style license
  - Used for neural network implementation and training
- **python-chess**: [https://python-chess.readthedocs.io/](https://python-chess.readthedocs.io/)
  - License: GPL-3.0
  - Used for chess move validation, board representation, and PGN parsing
- **NumPy**: [https://numpy.org/](https://numpy.org/)
  - License: BSD
  - Used for numerical operations and tensor manipulation
- **tqdm**: [https://tqdm.github.io/](https://tqdm.github.io/)
  - License: MIT/MPLv2.0
  - Used for progress bars during training and data processing
- **pandas**: [https://pandas.pydata.org/](https://pandas.pydata.org/)
  - License: BSD
  - Used for evaluation data analysis
- **matplotlib & seaborn**: [https://matplotlib.org/](https://matplotlib.org/) & [https://seaborn.pydata.org/](https://seaborn.pydata.org/)
  - License: BSD-style (matplotlib), BSD 3-Clause (seaborn)
  - Used for visualization of training results
- **zstandard**: [https://github.com/indygreg/python-zstandard](https://github.com/indygreg/python-zstandard)
  - License: BSD
  - Used for decompressing .zst PGN files

## External Code & Inspiration
- **AlphaZero General Framework**: Conceptual inspiration from AlphaZero-general architecture
  - Original: Surag Nair et al. (repo: https://github.com/suragnair/alpha-zero-general/tree/master)
  - Our implementation of Game, Coach, MCTS, and Arena classes follows the general AlphaZero-style architecture.
  - Core framework structure (Game base class, MCTS tree search, self-play loop) adapted from open-source AlphaZero

- **Stockfish Baseline Model**: General public source stockfish chess model used for evaluation measurement
    - Downloaded from: https://stockfishchess.org/
    - Used in: src\eval\against_stockfish.py and src\eval\total_eval.py

## AI Assistance
- Portions of code and documentation drafted with help from **ChatGPT** (OpenAI) and **GitHub Copilot**, then reviewed, edited, and integrated by the project team
- AI assistance used for: code structure suggestions, documentation writing, debugging help, sanity checks, and implementation guidance
    - Primarily in sections of the: encoding/decoding logic of AlphaZero, setting up the Gym environment, and running sanity checks/debugging after it was all said and done.