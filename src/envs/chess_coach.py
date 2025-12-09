from typing import Any, List, Tuple
from envs.chess_gym import GymChessEnv

class GymCoachAdapter:
    """
    Adapter that exposes the classic Game interface Coach expects while
    delegating to a GymChessEnv (which itself wraps the project's ChessGame).
    Use this adapter in place of the original ChessGame when creating Coach.
    """
    def __init__(self, env: GymChessEnv | None = None):
        self.env = env or GymChessEnv()
        self._game = self.env.game

    # board / game shape queries
    def getBoardSize(self) -> Tuple[int, int]:
        return self._game.getBoardSize()

    def getActionSize(self) -> int:
        return self._game.getActionSize()

    # core game API (stateless functions expected by Coach/MCTS)
    def getInitBoard(self) -> Any:
        return self._game.getInitBoard()

    def getNextState(self, board: Any, player: int, action: int) -> Tuple[Any, int]:
        return self._game.getNextState(board, player, action)

    def getValidMoves(self, board: Any, player: int) -> List[int]:
        return self._game.getValidMoves(board, player)

    def getGameEnded(self, board: Any, player: int) -> float:
        return self._game.getGameEnded(board, player)

    def getCanonicalForm(self, board: Any, player: int) -> Any:
        return self._game.getCanonicalForm(board, player)

    # optional helpers used by Coach/MCTS training/evaluation
    def getSymmetries(self, board: Any, pi: List[float]) -> List[Tuple[Any, List[float]]]:
        if hasattr(self._game, "getSymmetries"):
            return self._game.getSymmetries(board, pi)
        # fallback: no symmetries available
        return [(board, pi)]

    def stringRepresentation(self, board: Any) -> str:
        if hasattr(self._game, "stringRepresentation"):
            return self._game.stringRepresentation(board)
        return str(board)

    # convenience: expose the wrapped env for direct gym usage if desired
    @property
    def gym_env(self) -> GymChessEnv:
        return self.env