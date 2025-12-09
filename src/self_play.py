"""
Self-play training entry point for the Chess AlphaZero agent.

Usage:
    python src/train_self_play.py
    
    Or from repo root:
    python -m src.train_self_play
"""

import sys
import logging
from pathlib import Path
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.Coach import Coach
from core.args import ArgsAlphaZero
from chess_nnet.NNetWrapper import NNetWrapper
from envs.chess_coach import GymCoachAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def main():
    """
    Main entry point for self-play training.
    
    This script:
    1. Initializes the game adapter (wraps ChessGame via GymChessEnv)
    2. Creates a neural network wrapper
    3. Instantiates Coach with MCTS-guided self-play
    4. Runs the training loop
    """
    
    log.info("=" * 80)
    log.info("Starting Chess AlphaZero Self-Play Training")
    log.info("=" * 80)
    
    try:
        # Create training arguments
        args = ArgsAlphaZero()
        log.info(f"Training config:")
        log.info(f"  - MCTS Simulations per move: {args.numMCTSSims}")
        log.info(f"  - Training iterations: {args.numIters}")
        log.info(f"  - Episodes per iteration: {args.numEps}")
        log.info(f"  - Batch size: {args.batch_size}")
        log.info(f"  - Learning rate: {args.lr}")
        log.info(f"  - Checkpoint directory: {args.checkpoint}")
        
        # Initialize game (with Gym adapter wrapper)
        log.info("\nInitializing game environment...")
        game = GymCoachAdapter()
        log.info(f"  - Board size: {game.getBoardSize()}")
        log.info(f"  - Action space size: {game.getActionSize()}")
        
        # Initialize neural network
        log.info("\nInitializing neural network...")
        nnet = NNetWrapper(game)
        log.info(f"  - Model architecture: ChessResNet")
        log.info(f"  - Channels: {args.num_channels}")
        log.info(f"  - Residual blocks: {args.num_res_blocks}")
        log.info(f"  - CUDA available: {args.cuda}")
        
        # Create checkpoint directory if it doesn't exist
        Path(args.checkpoint).mkdir(parents=True, exist_ok=True)
        
        # Initialize Coach (combines MCTS + self-play + training)
        log.info("\nInitializing Coach (self-play orchestrator)...")
        coach = Coach(game, nnet, args)
        
        # Start training loop
        log.info("\n" + "=" * 80)
        log.info("Beginning self-play training loop")
        log.info("=" * 80 + "\n")
        
        coach.learn()
        
        log.info("\n" + "=" * 80)
        log.info("Training completed successfully!")
        log.info("=" * 80)
        
    except Exception as e:
        log.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
