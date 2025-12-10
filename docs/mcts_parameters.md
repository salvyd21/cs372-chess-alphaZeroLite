# MCTS Configuration

## Parameters
- `numMCTSSims`: Number of simulations per move (default: 50)
- `cpuct`: Exploration constant for UCB (default: 1.0)
- `temp`: Temperature for action selection
  - temp=1: Proportional to visit counts
  - temp=0: Greedy (max visit count)

## Search Process
1. Selection: Traverse tree using UCB formula
2. Expansion: Add new node for unexplored state
3. Evaluation: Query neural network for (policy, value)
4. Backup: Update Q-values and visit counts

## Storage
- `Qsa`: Q-values for state-action pairs
- `Nsa`: Visit counts for edges
- `Ps`: Neural network policy predictions