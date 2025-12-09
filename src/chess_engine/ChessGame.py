#built off the skeleton /core/Game.py from the general AlphaZero framework
import chess
import numpy as np
from .action_encoding import ACTION_SIZE
from .state_encoding import board_to_tensor


class ChessGame:
    """
    Chess implementation of the AlphaZero-General Game interface.

    Internal board representation: python-chess.Board
    Player: +1 or -1
      - In canonical form, +1 is always "white to move" from current player's POV.
    """
    def __init__(self):
        pass

    # Basic game info; board size and action size

    def getInitBoard(self):
        """Return initial chess board as chess.Board object"""
        return chess.Board()

    def getBoardSize(self):
        """Return (width, height) of the board"""
        return (8, 8)

    def getActionSize(self):
        """Return number of possible actions"""
        return ACTION_SIZE

    # State transitions

    def getNextState(self, board, player, action):
        """
        Execute action on board and return new board & next player.

        Args:
            board: chess.Board object
            player: 1 (White) or -1 (Black)
            action: integer action index

        Returns:
            (new_board, next_player)
        """
        from .action_encoding import decode_move_index

        move = decode_move_index(board, action)
        board.push(move)
        return board, -player

    # Valid moves

    def getValidMoves(self, board, player):
        """
        Return numpy array of valid move indices.

        Args:
            board: chess.Board object
            player: 1 or -1 (unused in chess, all moves valid regardless)

        Returns:
            numpy array of shape (ACTION_SIZE,) with 1.0 for valid moves, 0.0 else
        """
        from .action_encoding import encode_move_index

        valid_moves = np.zeros(ACTION_SIZE, dtype=np.float32)
        for move in board.legal_moves:
            action_idx = encode_move_index(board, move)
            if action_idx is not None:
                valid_moves[action_idx] = 1.0
        return valid_moves

    # Game termination

    def getGameEnded(self, board, player):
        """
        Check if game is over and return result.

        Returns:
            0 if game ongoing
            1 if player won
            -1 if player lost
            0.5 if draw (but return as 0 for simplicity)
        """
        if not board.is_game_over():
            return 0

        # Game is over
        if board.is_checkmate():
            # Current player is checkmated, so they lose
            return -1 if board.turn == (player == 1) else 1
        else:
            # Draw
            return 0

    # Canonical form & symmetries

    def getCanonicalForm(self, board, player):
        """
        Return canonical form of the board from the POV of 'player'.

        - If player == +1: board as-is (White's perspective)
        - If player == -1: mirror the board (Black's perspective becomes White's)
        """
        if player == 1:
            return board
        else:
            return board.mirror()

    # String representation (for MCTS hashing)

    def stringRepresentation(self, board):
        """Return FEN string representation of board"""
        return board.fen()
