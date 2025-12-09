import gymnasium as gym
from gymnasium import spaces
import numpy as np
from chess_engine.ChessGame import ChessGame
from chess_engine.state_encoding import board_to_tensor

class GymChessEnv(gym.Env):
    """
    Gymnasium wrapper around the project's ChessGame.
    Observations are canonical board tensors (12,8,8) (float32).
    Actions are discrete integers in [0, ACTION_SIZE).
    """
    def __init__(self, game: ChessGame = None):
        self.game = game or ChessGame()
        self.action_size = self.game.getActionSize()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(12, 8, 8), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_size)
        self._board = None
        self._player = 1

    def reset(self, *, seed=None, options=None):
        self._board = self.game.getInitBoard()
        self._player = 1
        obs = self.game.getCanonicalForm(self._board, self._player)
        return board_to_tensor(obs), {}

    def step(self, action):
        # action is assumed to be the flat index used by getNextState (canonical POV)
        next_board, next_player = self.game.getNextState(self._board, self._player, int(action))
        self._board = next_board
        self._player = next_player

        # canonical observation for the next current player
        obs_board = self.game.getCanonicalForm(self._board, self._player)
        obs = board_to_tensor(obs_board)

        result = self.game.getGameEnded(self._board, self._player)
        terminated = (result != 0)
        reward = float(result) if terminated else 0.0
        truncated = False
        return obs, reward, terminated, truncated, {}

    def render(self):
        print(self._board)

    def close(self):
        pass
