# imports
...

class ChessNNet(torch.nn.Module):
    def __init__(self, game, args):
        ...
    def forward(self, x):
        # returns (policy_logits, value)
