import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
from sklearn.model_selection import KFold
import scipy
import matplotlib.pyplot as plt
import sklearn

# Critic Network
class Critic(nn.Module):
    def __init__(self, channels=1, hidden_dim=128):
        super(Critic, self).__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim

        # Convolutional branch for time series data
        self.conv_branch = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim // 4, 1, kernel_size=5, stride=1, padding=2),
        )

        self.conv_branch.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.3)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, target):
        return self.conv_branch(target)


class ConvDenoiser(nn.Module):
    def __init__(self, channels=1, hidden_dim=128):
        super(ConvDenoiser, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Conv2d(channels, hidden_dim // 4, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )

        self.flag_condition = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv2d(channels, hidden_dim // 4, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2 * hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dim // 4, channels, kernel_size=3, padding=1),
        )

        self.encoder.apply(self.init_weights)
        self.flag_condition.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.3)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x, flag):
        encoded = self.encoder(x)
        flag = self.flag_condition(flag)
        combined = torch.cat([encoded, flag], dim=1)
        decoded = self.decoder(combined)
        return decoded

def evaluate_model(generator, critic, loader, device):
    total_c_loss = 0
    total_g_loss = 0
    total_mse_loss = 0

    mse_criterion = nn.MSELoss(reduction="mean")

    generator.eval()
    critic.eval()

    with torch.no_grad():
        for batch in loader:
            X = batch[0].to(device)
            # Fill nans with batch average plus noise
            noise = torch.randn_like(X)
            was_nan = torch.isnan(X).to(device)
            X[was_nan] = torch.mean(X[~was_nan]) + noise[was_nan].to(device) * torch.std(X[~was_nan])
            X_pred = generator(X, (~was_nan).type(torch.float32))
            validity = critic(X_pred)
            c_loss = torch.mean(validity[was_nan]) - torch.mean(validity[~was_nan])
            g_loss = -torch.mean(validity[was_nan])
            mse_loss = mse_criterion(X_pred[~was_nan], X[~was_nan])

            total_c_loss += c_loss
            total_g_loss += g_loss
            total_mse_loss += mse_loss


    batch_avg_mse_loss = total_mse_loss / len(loader)
    batch_avg_g_loss = total_g_loss / len(loader)
    batch_avg_c_loss = total_c_loss / len(loader)

    return batch_avg_mse_loss, batch_avg_c_loss, batch_avg_g_loss

def train_denoiser_adversarially(
    generator,
    critic,
    training_dataloader,
    testing_dataloader,
    num_epochs,
    device,
    generator_learning_rate=8e-5,
    critic_learning_rate=1e-6
):

    g_optimizer = optim.Adam(generator.parameters(), lr=generator_learning_rate, betas=(0.8, 0.9), weight_decay=1e-5)
    c_optimizer = optim.Adam(critic.parameters(), lr=critic_learning_rate, betas=(0.8, 0.9), weight_decay=1e-5)
    mse_criterion = nn.MSELoss(reduction="mean")

    for epoch in range(num_epochs):
        generator.train()
        critic.train()
        cum_generator_loss = 0
        cum_critic_loss = 0
        cum_mse_loss = 0

        for i, batch in enumerate(tqdm.tqdm(training_dataloader)):

            X = batch[0].to(device)

            # Fill nans with batch average plus noise
            noise = torch.randn_like(X)
            was_nan = torch.isnan(X).to(device)
            X[was_nan] = torch.mean(X[~was_nan]) + noise[was_nan].to(device) * torch.std(
                X[~was_nan]
            )

            # Train Critic
            X_pred_c = generator(X, (~was_nan).type(torch.float32))
            c_optimizer.zero_grad()
            validity = critic(X_pred_c)

            c_loss = torch.mean(validity[was_nan]) - torch.mean(validity[~was_nan])

            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.1)
            c_optimizer.step()
            cum_critic_loss += c_loss.item()

            # Train Generator
            g_optimizer.zero_grad()
            X_pred_g = generator(X, (~was_nan).type(torch.float32))

            if i % 5 == 0:
                validity = critic(X_pred_g)
                g_loss = -torch.mean(validity[was_nan])
                g_loss.backward()
                cum_generator_loss += g_loss.item()
            else:
                mse_loss = mse_criterion(X_pred_g[~was_nan], X[~was_nan])
                mse_loss.backward()
                cum_mse_loss += mse_loss.item()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=0.10)
            g_optimizer.step()

        avg_generator_loss = cum_generator_loss / ((1 / 5) * len(training_dataloader))
        avg_critic_loss = cum_critic_loss / (len(training_dataloader))
        avg_mse_loss = cum_mse_loss / ((4 / 5) * len(training_dataloader))

        print("Testing Model on Test Data...\n")
        test_avg_MSE_loss, test_avg_c_loss, test_avg_g_loss = evaluate_model(generator,
                                                                             critic,
                                                                             testing_dataloader,
                                                                             device)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}]\n"
            f"Averages:\n"
            f"Training Critic Loss: {avg_critic_loss:.4f}\n"
            f"Trainin Generator Loss: {avg_generator_loss:.4f}\n"
            f"Training MSE : {avg_mse_loss:4f}\n"
            f"Test Critic Loss: {test_avg_c_loss:.4f}\n"
            f"Test Generator Loss: {test_avg_g_loss:.4f}\n"
            f"Test MSE : {test_avg_MSE_loss:4f}\n"
        )

def cross_validate_model(
    dataset,
    denoiser_class,
    critic_class,
    n_splits=5,
    batch_size=32,
    num_epochs=5,
    learning_rate=0.001,
    denoiser_learning_rate=8e-5,
    critic_learning_rate=1e-6,
    num_channels=1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []
    trained_models = []
    transformers = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        print(f"\nFold {fold + 1}/{n_splits}\n")

        train_set = dataset[train_idx]
        test_set = dataset[test_idx]

        averaged_train_for_box_cox_fit = np.nanmean(train_set, axis=0).flatten()

        PT = sklearn.preprocessing.PowerTransformer(method='box-cox', standardize=True)
        PT.fit(averaged_train_for_box_cox_fit[:, np.newaxis])

        transformed_train_set = PT.transform(train_set.reshape(train_set.shape[0] * train_set.shape[1] * train_set.shape[2], 1))
        train_set = transformed_train_set.reshape(train_set.shape[0], train_set.shape[1], train_set.shape[2])

        transformed_test_set = PT.transform(test_set.reshape(test_set.shape[0] * test_set.shape[1] * test_set.shape[2], 1))
        test_set = transformed_test_set.reshape(test_set.shape[0], test_set.shape[1], test_set.shape[2])

        transformers.append(PT)

        print("Creating Data Loaders")
        # Create data loaders
        train_loader = create_sample_dataloader(train_set, batch_size=16)
        test_loader = create_sample_dataloader(test_set, batch_size=16)

        print("Creating Denoiser and Critic")
        denoiser = denoiser_class(channels=num_channels).to(device)
        critic = critic_class().to(device)

        print("Training...\n")
        train_denoiser_adversarially(denoiser,
                                     critic,
                                     train_loader,
                                     test_loader,
                                     num_epochs,
                                     device,
                                     generator_learning_rate=denoiser_learning_rate,
                                     critic_learning_rate=critic_learning_rate)

        print("Testing Final Model on Training Data...\n")
        train_avg_MSE_loss, train_avg_c_loss, train_avg_g_loss = evaluate_model(denoiser,
                                                                                critic,
                                                                                train_loader,
                                                                                device)

        print("Testing Final Model on Test Data...\n")
        test_avg_MSE_loss, test_avg_c_loss, test_avg_g_loss = evaluate_model(denoiser,
                                                                             critic,
                                                                             test_loader,
                                                                             device)

        print("Results:\n")
        print(
            f"Train MSE : {train_avg_MSE_loss}\n"
            f"Train g_loss : {train_avg_g_loss}\n"
            f"Train c_loss : {train_avg_c_loss}\n"
            f"Test MSE : {test_avg_MSE_loss}\n"
            f"Test g_loss : {test_avg_g_loss}\n"
            f"Test c_loss : {test_avg_c_loss}\n"
        )

        fold_results.append({"Fold": fold + 1,
                             "Train_MSE" : train_avg_MSE_loss,
                             "Train_g_loss" : train_avg_g_loss,
                             "Train_c_loss": train_avg_c_loss,
                             "Test_mse" : test_avg_MSE_loss,
                             "Test_g_loss" : test_avg_g_loss,
                             "Test_c_loss": test_avg_c_loss})

        trained_models.append({"denoiser": denoiser, "critic" : critic})

    # Calculate average metrics across folds
    fold_avg_test_MSE = torch.mean([result["Test_mse"] for result in fold_results])
    fold_avg_test_g_loss = torch.mean([result["Test_g_loss"] for result in fold_results])
    fold_avg_test_c_loss = torch.mean([result["Test_c_loss"] for result in fold_results])

    print(f"\nFold Average Test MSE : {fold_avg_test_MSE:.4f}\n")
    print(f"Fold Average Test g_loss : {fold_avg_test_g_loss:.4f}\n")
    print(f"Fold Average Test c_loss : {fold_avg_test_c_loss:4f}\n")

    return fold_results, trained_models, transformers


def create_sample_dataloader(images, batch_size=32):
    """Create a DataLoader from a numpy array or torch tensor of images."""
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).float()
    if images.dim() == 3:  # Add channel dimension if needed
        images = images.unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(images)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
