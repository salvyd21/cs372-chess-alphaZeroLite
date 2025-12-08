class ChessMoveDataset(Dataset):
    ...

def train_supervised():
    model = ChessNNet(...)
    loader = DataLoader(...)
    optimizer = ...
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(E):
        for X, y in loader:
            ...
