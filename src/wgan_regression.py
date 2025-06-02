import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Custom Dataset for time series and conditional coordinates
class TimeSeriesDataset(Dataset):
    def __init__(self, time_series_data, coordinates, targets):
        """
        time_series_data: List of tensors, each of shape (num_vectors, time_steps)
        coordinates: List of tensors, each of shape (num_conditions,)
        targets: List of tensors, each of shape (output_dim,)
        """
        self.time_series_data = time_series_data
        self.coordinates = coordinates
        self.targets = targets
        assert len(time_series_data) == len(coordinates) == len(targets)

    def __len__(self):
        return len(self.time_series_data)

    def __getitem__(self, idx):
        return self.time_series_data[idx], self.coordinates[idx], self.targets[idx]

# Generator Network
class Generator(nn.Module):
    def __init__(self, noise_dim, condition_dim, output_dim, hidden_dim, num_vectors):
        super(Generator, self).__init__()
        self.num_vectors = num_vectors
        self.output_dim = output_dim

        # Convolutional branch for time series data
        self.conv_branch = nn.Sequential(
            nn.Conv1d(num_vectors, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim // 2)
        )

        # Fully connected branch for conditional coordinates
        self.fc_branch = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2)
        )

        # Noise input
        self.noise_fc = nn.Linear(noise_dim, hidden_dim // 2)

        # Combined branch (input size: hidden_dim/2 from conv + hidden_dim/2 from fc + hidden_dim/2 from noise)
        self.combined = nn.Sequential(
            nn.Linear((hidden_dim // 2) * 3, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, noise, time_series, conditions):
        # Process time series data: (batch, num_vectors, time_steps)
        conv_out = self.conv_branch(time_series)
        conv_out = conv_out.mean(dim=2)  # Global average pooling: (batch, hidden_dim/2)

        # Process conditional coordinates: (batch, condition_dim)
        cond_out = self.fc_branch(conditions)  # (batch, hidden_dim/2)

        # Process noise: (batch, noise_dim)
        noise_out = self.noise_fc(noise)  # (batch, hidden_dim/2)

        # Combine features
        combined = torch.cat([conv_out, cond_out, noise_out], dim=1)  # (batch, 3 * hidden_dim/2)
        output = self.combined(combined)  # (batch, output_dim)
        return output

# Critic Network
class Critic(nn.Module):
    def __init__(self, target_dim, condition_dim, hidden_dim, num_vectors):
        super(Critic, self).__init__()
        self.num_vectors = num_vectors
        self.target_dim = target_dim

        # Convolutional branch for time series data
        self.conv_branch = nn.Sequential(
            nn.Conv1d(num_vectors, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2)
        )

        # Fully connected branch for conditional coordinates
        self.fc_branch = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2)
        )

        # Target input (regression targets)
        self.target_fc = nn.Linear(target_dim, hidden_dim // 2)

        # Combined branch (input size: hidden_dim/2 from conv + hidden_dim/2 from fc + hidden_dim/2 from target)
        self.combined = nn.Sequential(
            nn.Linear((hidden_dim // 2) * 3, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)  # Output a single score for Wasserstein loss
        )

    def forward(self, target, time_series, conditions):
        # Process time series data: (batch, num_vectors, time_steps)
        conv_out = self.conv_branch(time_series)
        conv_out = conv_out.mean(dim=2)  # Global average pooling: (batch, hidden_dim/2)

        # Process conditional coordinates: (batch, condition_dim)
        cond_out = self.fc_branch(conditions)  # (batch, hidden_dim/2)

        # Process target: (batch, target_dim)
        target_out = self.target_fc(target)  # (batch, hidden_dim/2)

        # Combine features
        combined = torch.cat([conv_out, cond_out, target_out], dim=1)  # (batch, 3 * hidden_dim/2)
        output = self.combined(combined)  # (batch, 1)
        return output

# Gradient Penalty
def compute_gradient_penalty(critic, real_samples, fake_samples, time_series, conditions, device):
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand(real_samples.size())

    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    critic_interpolates = critic(interpolates, time_series, conditions)

    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Training Loop
def train_wgan_gp(generator, critic, dataloader, noise_dim, condition_dim, num_epochs, device, lambda_gp=10):
    g_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    c_optimizer = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))
    n_critic = 5  # Number of critic updates per generator update

    for epoch in range(num_epochs):
        for i, (time_series, conditions, real_targets) in enumerate(dataloader):
            batch_size = real_targets.size(0)
            time_series, conditions, real_targets = time_series.to(device), conditions.to(device), real_targets.to(device)

            # Train Critic
            for _ in range(n_critic):
                noise = torch.randn(batch_size, noise_dim, device=device)
                fake_targets = generator(noise, time_series, conditions)

                c_optimizer.zero_grad()
                real_validity = critic(real_targets, time_series, conditions)
                fake_validity = critic(fake_targets.detach(), time_series, conditions)

                gradient_penalty = compute_gradient_penalty(critic, real_targets, fake_targets.detach(), time_series, conditions, device)
                c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

                c_loss.backward()
                c_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_targets = generator(noise, time_series, conditions)
            g_loss = -torch.mean(critic(fake_targets, time_series, conditions))

            g_loss.backward()
            g_optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] "
                      f"Critic Loss: {c_loss.item():.4f} Generator Loss: {g_loss.item():.4f}")
                
    return generator, critic

# Example Usage
if __name__ == "__main__":
    # Hyperparameters
    noise_dim = 100  # Noise dimension
    condition_dim = 10  # Number of conditional coordinates
    output_dim = 5  # Regression output dimension
    hidden_dim = 128
    num_vectors = 3  # Number of time series vectors
    time_steps = 50  # Length of each time series
    batch_size = 32
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    generator = Generator(noise_dim, condition_dim, output_dim, hidden_dim, num_vectors).to(device)
    critic = Critic(output_dim, condition_dim, hidden_dim, num_vectors).to(device)

    # Dummy dataset (replace with your data)
    time_series_data = [torch.randn(num_vectors, time_steps) for _ in range(1000)]
    coordinates = [torch.randn(condition_dim) for _ in range(1000)]
    targets = [torch.randn(output_dim) for _ in range(1000)]
    dataset = TimeSeriesDataset(time_series_data, coordinates, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    trained_generator, trained_critic = train_wgan_gp(generator, critic, dataloader, noise_dim, condition_dim, num_epochs, device)
    
    