"""
Configuration/hyperparameters for Coach training.
"""

class ArgsAlphaZero:
    # MCTS 
    numMCTSSims = 25        # Number of simulations per MCTS search
    cpuct = 1.0             # UCB exploration constant
    
    # Training Loop
    numIters = 10           # Number of training iterations
    numEps = 100            # Number of self-play episodes per iteration
    tempThreshold = 15      # Moves before temperature drops to 0 (for policy sampling)
    maxlenOfQueue = 200000  # Max number of examples to keep in training history
    numItersForTrainExamplesHistory = 20  # Keep examples from last N iterations
    
    # Arena (competitive evaluation)
    arenaCompare = 40       # Number of games to play in arena
    updateThreshold = 0.55  # New net must win > 55% to replace old net
    
    # Neural Network
    num_channels = 128      # Convolutional channels
    num_res_blocks = 5      # Number of residual blocks
    dropout = 0.3
    batch_size = 64
    epochs = 10
    lr = 0.001              # Learning rate
    l2 = 0.0001             # L2 regularization
    cuda = True             # Use GPU if available
    
    # File I/O 
    checkpoint = "models"   # Directory to save model checkpoints
    load_folder_file = None # (Optional) Path to resume from checkpoint
