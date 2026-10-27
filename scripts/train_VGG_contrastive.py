import torch
from model import MeanPooledVGG

def main():
    # Create dummy input: batch of 8 samples, 5 channels, 192x288 resolution
    dummy_input = torch.randn(8, 5, 192, 288)

    # Initialize model
    model = MeanPooledVGG(in_channels=5, projection_dim=128)

    # Forward pass
    output = model(dummy_input)

    # Output shape should be (8, 128)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()
