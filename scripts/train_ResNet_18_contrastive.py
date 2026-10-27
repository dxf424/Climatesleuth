import torch
from model import MeanPooledResNet18

def main():
    # Create dummy input with 5 channels, 192x288 resolution
    dummy_input = torch.randn(8, 5, 192, 288)  # Batch size of 8

    # Initialize model
    model = MeanPooledResNet18(in_channels=5, projection_dim=128)

    # Forward pass
    output = model(dummy_input)

    # Print output shape
    print("Output shape:", output.shape)  # Should be (8, 128)

if __name__ == "__main__":
    main()
