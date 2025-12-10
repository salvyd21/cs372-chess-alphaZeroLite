# Chess Neural Network Architecture

## Model: ChessResNet

### Input
- Shape: (batch, 12, 8, 8)
- 12 channels: 6 piece types × 2 colors
  - Channels 0-5: White pieces (P, N, B, R, Q, K)
  - Channels 6-11: Black pieces (P, N, B, R, Q, K)

### Architecture
1. **Initial Convolution Block**
   - Conv2d(12 → 128 channels, 3×3 kernel, padding=1)
   - BatchNorm2d
   - ReLU

2. **Residual Tower** (5 blocks by default)
   - Each ResBlock: Conv3×3 → BN → ReLU → Conv3×3 → BN → Add → ReLU

3. **Policy Head**
   - Conv2d(128 → 2, 1×1 kernel)
   - Flatten → Linear(128 → 4672)
   - LogSoftmax
   - Output: log probabilities over 4,672 actions

4. **Value Head**
   - Conv2d(128 → 1, 1×1 kernel)
   - Flatten → Linear(64 → 256) → ReLU → Linear(256 → 1)
   - Tanh
   - Output: position evaluation in [-1, 1]

### Hyperparameters (default)
- Channels: 128
- ResBlocks: 5
- Learning rate: 0.001
- Dropout: 0.3
- L2 regularization: 0.0001