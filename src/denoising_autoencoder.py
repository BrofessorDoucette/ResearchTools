import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ConvDenoiser(nn.Module):
    def __init__(self, channels=1):
        super(ConvDenoiser, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_denoiser(model, train_loader, num_epochs=10, noise_factor=0, learning_rate=0.001, device="cuda"):
    """Train the denoising autoencoder."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction="mean")  # No reduction to handle NaN masking

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            X = batch[0].to(device)

            mask = torch.isnan(X)  # Mask for non-NaN values
            noise = torch.randn_like(X) * noise_factor
            X[mask] = noise[mask]

            # Forward pass
            optimizer.zero_grad()
            X_pred = model(X)

            # Compute loss only for non-NaN values
            loss = criterion(X_pred[mask], X[mask])
            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


def create_sample_dataloader(images, batch_size=32):
    """Create a DataLoader from a numpy array or torch tensor of images."""
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).float()
    if images.dim() == 3:  # Add channel dimension if needed
        images = images.unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(images)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Example usage
if __name__ == "__main__":
    # Sample data (replace with your own dataset)
    sample_images = torch.randn(100, 1, 64, 64)  # 100 images, 1 channel, 64x64
    sample_images[0, :, :32, :32] = np.nan  # Add some NaN values for testing

    # Create DataLoader
    train_loader = create_sample_dataloader(sample_images, batch_size=32)

    # Initialize and train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvDenoiser(channels=1)
    train_denoiser(model, train_loader, num_epochs=10, device=device)
