import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Generator Network
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for noise
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Single real number output
        )

    def forward(self, x, noise):
        # Concatenate input features with noise
        input_with_noise = torch.cat((x, noise), dim=1)
        return self.model(input_with_noise)


# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for target value
            nn.LeakyReLU(0.005),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.005),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output probability of real data
        )

    def forward(self, x, y):
        # Concatenate input features with target value
        input_with_target = torch.cat((x, y), dim=1)
        return self.model(input_with_target)


# GAN Regression Model
class GANRegression:
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        d_lr=0.0002,
        g_lr=0.0002,
        betas=(0.5, 0.90),
        weight_decay=1e-4,
        epochs_per_output=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        print(f"using device : {device}")
        self.device = torch.device(device)
        self.generator = Generator(input_dim, hidden_dim).to(self.device)
        self.discriminator = Discriminator(input_dim, hidden_dim).to(self.device)

        self.g_optimizer = optim.Adam(
            self.generator.parameters(), lr=g_lr, betas=betas, weight_decay=weight_decay
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=d_lr, betas=betas, weight_decay=weight_decay
        )

        self.epochs_per_output = epochs_per_output

        self.bce_loss = nn.BCELoss()

    def train(self, X, y, epochs=100, batch_size=32):
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).reshape(-1, 1).to(self.device)

        n_samples = X.shape[0]
        n_batches = n_samples // batch_size

        avg_g_losses = []
        avg_d_losses = []
        avg_n_losses = []

        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            g_losses = []
            d_losses = []
            n_losses = []

            for b in range(0, n_batches):
                # Get batch
                batch_X = X_shuffled[b * batch_size : (b * batch_size + batch_size)]
                batch_y = y_shuffled[b * batch_size : (b * batch_size + batch_size)]

                # Train Discriminator
                self.d_optimizer.zero_grad()

                real_labels = torch.ones_like(batch_y).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                # Real data
                d_real = self.discriminator(batch_X, batch_y)

                # Fake data from generator
                noise = torch.abs(torch.randn(batch_size, 1)).to(self.device)
                fake_y = self.generator(batch_X, noise)
                d_fake = self.discriminator(batch_X, fake_y)

                d_real_and_fake = torch.cat((d_real, d_fake), dim=0)
                labels_real_and_fake = torch.cat((real_labels, fake_labels), dim=0)

                num_rows = d_real_and_fake.size(0)
                shuffled_indices = torch.randperm(num_rows)
                d_real_and_fake = d_real_and_fake[shuffled_indices]
                labels_real_and_fake = labels_real_and_fake[shuffled_indices]

                d_loss = self.bce_loss(d_real_and_fake, labels_real_and_fake)
                d_loss.backward()
                self.d_optimizer.step()

                # Train Generator

                self.g_optimizer.zero_grad()
                noise = torch.abs(torch.randn(batch_size, 1)).to(self.device)
                fake_y = self.generator(batch_X, noise)
                g_output = self.discriminator(batch_X, fake_y)
                g_loss = self.bce_loss(g_output, real_labels)

                # Add regression loss to encourage correct values
                huber_loss = nn.HuberLoss()(fake_y, batch_y)
                g_total_loss = g_loss + huber_loss
                g_total_loss.backward()
                self.g_optimizer.step()

                d_losses.append(d_loss.item())
                n_losses.append(huber_loss.item())
                g_losses.append(g_loss.item())

            if (epoch + 1) % self.epochs_per_output == 0:
                print(
                    f"""Epoch [{epoch + 1} / {epochs}] \nD Loss: ({np.nanmean(d_losses):.4f} +/- {np.nanstd(d_losses):.4f}) N Loss: ({np.nanmean(n_losses):.4f} +/- {np.nanstd(n_losses):.4f}) G Loss: ({np.nanmean(g_losses):.4f} +/- {np.nanstd(g_losses):.4f})\n"""
                )

            avg_d_losses.append(np.nanmean(d_losses))
            avg_n_losses.append(np.nanmean(n_losses))
            avg_g_losses.append(np.nanmean(g_losses))

        return avg_d_losses, avg_n_losses, avg_g_losses

    def GENERATE(self, X):
        X = torch.FloatTensor(X).to(self.device)
        noise = torch.abs(torch.randn(X.shape[0], 1)).to(self.device)
        with torch.no_grad():
            y_pred = self.generator(X, noise)
        return y_pred.cpu().numpy().flatten()

    def DISCRIM(self, X, y):
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).reshape(-1, 1).to(self.device)
        with torch.no_grad():
            y_pred = self.discriminator(X, y)
        return y_pred.cpu().numpy().flatten()


# Example usage
if __name__ == "__main__":
    # Generate sample data
    X = np.random.rand(1000, 5)  # 1000 samples, 5 features
    y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.normal(0, 0.1, 1000)  # Example target

    # Initialize and train model
    model = GANRegression(input_dim=5)
    model.train(X, y, epochs=100, batch_size=32)

    # Make predictions
    predictions = model.predict(X[:10])
    print("Sample predictions:", predictions)
