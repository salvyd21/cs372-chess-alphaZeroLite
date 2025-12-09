import logging
import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)
        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        Args:
            canonicalBoard: chess.Board object
            temp: temperature for exploration
        
        Returns:
            probs: numpy array of shape (ACTION_SIZE,)
        """
        # Run MCTS simulations
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        # Get the string representation for lookup
        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = np.zeros(self.game.getActionSize(), dtype=np.float32)  # ← Use numpy array
            probs[bestA] = 1
            return probs

        counts = np.array(counts, dtype=np.float32)  # ← Convert to numpy
        counts = counts ** (1. / temp)
        counts_sum = float(np.sum(counts))
        probs = counts / counts_sum
        return probs  # ← Now returns numpy array


    def search(self, canonicalBoard):
        """
        Args:
            canonicalBoard: chess.Board object
        
        Returns:
            v: value of the current board position
        """
        from chess_engine.state_encoding import board_to_tensor
        
        s = self.game.stringRepresentation(canonicalBoard)

        # Check if game ended
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # Terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # Leaf node - evaluate with neural network
            board_tensor = board_to_tensor(canonicalBoard).astype(np.float32)
            self.Ps[s], v = self.nnet.predict(board_tensor)
            
            # Get and store valid moves
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Vs[s] = valids  # ← STORE valid moves here
            
            # Mask invalid moves in policy
            self.Ps[s] = self.Ps[s] * valids
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
            else:
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Ns[s] = 0
            return -v

        # Now safe to access self.Vs[s]
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # Pick action with highest UCB score
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * np.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * np.sqrt(self.Ns[s] + EPS)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard.copy(), 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
