#built off the skeleton /core/Game.py from the general AlphaZero framework
import numpy as np
import chess
from .action_encoding import (
    ACTION_SIZE,
    encode_move_index,
    decode_move_index,
)
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
        """Return initial board state."""
        return chess.Board()

    def getBoardSize(self):
        """Return board spatial dimensions (x, y)."""
        return (8, 8)

    def getActionSize(self):
        """Total number of possible actions in the flat action space."""
        return ACTION_SIZE

    # State transitions

    def getNextState(self, board, player, action):
        """
        Given a board (not necessarily canonical), a player (+1/-1),
        and an action index, return (next_board, next_player).

        For MCTS, this will typically be called with:
          - board = canonicalBoard (from getCanonicalForm)
          - player = 1
        """
        # Work on a copy to avoid side effects
        b = board.copy()

        # Decode action as a move on the canonical view of this player
        # Canonical view means player == 1 (White)-> board as-is, player == -1 (Black)-> mirrored
        canon_b = self.getCanonicalForm(b, player)
        move = decode_move_index(canon_b, action)

        if move not in canon_b.legal_moves:
            raise ValueError(f"Decoded illegal move {move} from action {action}")

        canon_b.push(move)

        # Now we need to convert canon_b back to a "neutral" board
        # where pieces/colors are in normal coordinates and the side to move is correct
        next_player = -player

        if next_player == 1:
            # Player +1 is the one to move; canonical board already has +1 as White
            next_board = canon_b
        else:
            # Player -1 is to move; we invert canonicalization:
            # getCanonicalForm(board, -1) = board.mirror()
            # so original board = canon_b.mirror()
            next_board = canon_b.mirror()

        return next_board, next_player

    # Valid moves

    def getValidMoves(self, board, player):
        """
        Return a binary vector of length ACTION_SIZE indicating valid moves.
        Uses the canonical view of (board, player).
        """
        canon_b = self.getCanonicalForm(board, player)
        valid_moves = np.zeros(self.getActionSize(), dtype=np.uint8)

        for move in canon_b.legal_moves:
            try:
                idx = encode_move_index(canon_b, move)
                valid_moves[idx] = 1
            except ValueError:
                # Skip moves that can't be encoded (shouldn't happen)
                pass
        return valid_moves

    # Game termination

    def getGameEnded(self, board, player):
        """
        Returns:
            0       if game not ended
            +1      if 'player' has won
            -1      if 'player' has lost
            small val   (1e-4) if draw

        We use canonical form so 'player' is always white to move.
        """
        canon_b = self.getCanonicalForm(board, player)

        if not canon_b.is_game_over():
            return 0

        result = canon_b.result()  # "1-0", "0-1", or "1/2-1/2"
        if result == "1-0":
            return 1 # 'Player' (White) Wins
        elif result == "0-1":
            return -1 # 'Player' (White) Loses -> Black Wins
        else:
            return 1e-4  # Draw


    # Canonical form & symmetries

    def getCanonicalForm(self, board, player):
        """
        Return canonical form of the board from the POV of 'player'.

        - If player == +1: board should represent them as White.
        - If player == -1: we mirror the board so that, from the POV of
          the current player, they look like White.

        In all cases, canonical board has white-to-move.
        """
        b = board.copy()

        if player == 1:
            # Ensure it's White to move in canonical representation
            b.turn = chess.WHITE
            return b
        else:
            # Mirror: swap colors, flip orientation, flip which side moves
            b = b.mirror()
            # Ensure white moves in the canonical form
            b.turn = chess.WHITE
            return b

    def getSymmetries(self, board, pi):
        """
        Data augmentation. For now: identity only.
        You could add flips/rotations if you also transform pi accordingly.
        """
        return [(board, pi)]

    # String representation (for MCTS hashing)

    def stringRepresentation(self, board):
        """
        Return a unique, hashable string for a board state.
        """
        return board.fen()
