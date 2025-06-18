import torch
import torch.nn as nn
import torch.optim as optim
torch.set_float32_matmul_precision("high")

import numpy as np
import tqdm
from sklearn.model_selection import GroupKFold
import os


# Custom Dataset for time series and conditional coordinates
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X_conditional, X_convolutional, y):

        self.X_conditional = X_conditional.astype(np.float32)
        self.X_convolutional = X_convolutional.astype(np.float32)
        self.y = y.astype(np.float32)
        assert len(X_conditional) == len(X_convolutional) == len(y)

    def __len__(self):
        return len(self.X_conditional)

    def __getitem__(self, idx):
        return self.X_conditional[idx, :], self.X_convolutional[idx, None, :], self.y[idx, None]



class BiasGenerator(nn.Module):

    def __init__(self, input_dim=1, hidden_dim=64, noise_dim=2):
        super(BiasGenerator, self).__init__()

        self.cond = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
        )

        # Convolutional branch for time series data
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 4, kernel_size=7, stride=1, padding=3),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.noise = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Dropout(0.1),
        )

        self.combined = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, X_cond, X_conv, noise):

        cond_features = self.cond(X_cond)
        conv_features = self.conv(X_conv)
        noise_features = self.noise(noise)
        
        return self.combined(torch.cat([cond_features, conv_features, noise_features], dim=1))

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.3)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

# Neural network model predicting mean and log-variance
class VarianceGenerator(nn.Module):

    def __init__(self, input_dim=1, hidden_dim=64):
        super(VarianceGenerator, self).__init__()
        
        self.cond = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2)
        )

        self.mean = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.LeakyReLU(0.2),
        )

        self.combined = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
        self.cond.apply(self.init_weights)
        self.mean.apply(self.init_weights)


    def forward(self, X_cond, pred_mean):

        cond_features = self.cond(X_cond)
        mean_features = self.mean(pred_mean)

        return self.combined(torch.cat([cond_features, mean_features], dim=1))

    def init_weights(self, m):
        if isinstance(m, (nn.Linear)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.3)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

def get_sampler_weights(targets):

    probabilities = np.ones_like(targets)

    probabilities = 1 / np.exp(4 * (targets / np.max(targets)))
    
    weights = 1 / probabilities
    weights = (weights / weights.sum())

    return torch.from_numpy(weights).type(torch.float32)


def evaluate_model(bias_model,
                   var_model,
                   gaussian_nll,
                   loader,
                   device):

    total_mse_loss_zero = torch.as_tensor(0).type(torch.float32).to(device)
    total_nll_loss_zero = torch.as_tensor(0).type(torch.float32).to(device)
    total_var_loss_zero = torch.as_tensor(0).type(torch.float32).to(device)

    total_mse_loss_nonzero = torch.as_tensor(0).type(torch.float32).to(device)
    total_nll_loss_nonzero = torch.as_tensor(0).type(torch.float32).to(device)
    total_var_loss_nonzero = torch.as_tensor(0).type(torch.float32).to(device)

    num_batches_non_zero = 0
    num_batches_zero = 0

    with torch.no_grad():

        bias_model.eval()
        var_model.eval()
        mse_criterion = torch.nn.MSELoss()

        for batch in loader:

            X_conditional, X_convolutional, y = batch

            non_zero_y = y != 0
            zero_y = y == 0

            X_conditional = X_conditional.to(device).type(torch.float32)
            X_convolutional = X_convolutional.to(device).type(torch.float32)
            y = y.to(device).type(torch.float32)

            if torch.any(non_zero_y):

                pred_mean = bias_model(X_conditional[non_zero_y.flatten(), :], X_convolutional[non_zero_y.flatten(), :, :], torch.randn(size=(len(y[non_zero_y.flatten()]), 2)).to(device))
                pred_var = var_model(X_conditional[non_zero_y.flatten(), :], pred_mean)
                
                nll_loss = gaussian_nll(pred_mean, y[non_zero_y.flatten(), :], pred_var)
                mse_loss = mse_criterion(pred_mean, y[non_zero_y.flatten(), :])
                var_loss = torch.mean(pred_var + 1e-9)

                total_nll_loss_nonzero += nll_loss
                total_mse_loss_nonzero += mse_loss
                total_var_loss_nonzero += var_loss
                num_batches_non_zero += 1

            if torch.any(zero_y):
                pred_mean = bias_model(X_conditional[zero_y.flatten(), :], X_convolutional[zero_y.flatten(), :, :], torch.randn(size=(len(y[zero_y.flatten()]), 2)).to(device))
                pred_var = var_model(X_conditional[zero_y.flatten(), :], pred_mean)
                
                nll_loss = gaussian_nll(pred_mean, y[zero_y.flatten(), :], pred_var)
                mse_loss = mse_criterion(pred_mean, y[zero_y.flatten(), :])
                var_loss = torch.mean(pred_var + 1e-9)

                total_nll_loss_zero += nll_loss
                total_mse_loss_zero += mse_loss
                total_var_loss_zero += var_loss

                num_batches_zero += 1

    avg_rmse_loss_nonzero = np.sqrt(total_mse_loss_nonzero.item() / num_batches_non_zero)
    avg_nll_loss_nonzero = total_nll_loss_nonzero.item() / num_batches_non_zero
    avg_var_loss_nonzero = total_var_loss_nonzero.item() / num_batches_non_zero


    #avg_rmse_loss_zero = np.sqrt(total_mse_loss_zero.item() / num_batches_zero)
    #avg_nll_loss_zero = total_nll_loss_zero.item() / num_batches_zero
    #avg_var_loss_zero = total_var_loss_zero.item() / num_batches_zero

    return avg_rmse_loss_nonzero, avg_var_loss_nonzero, avg_nll_loss_nonzero


def cross_validate_model(
    X_conditional,
    X_convolutional,
    y,
    g_data,
    bias_model_class,
    variance_model_class,
    bias_learning_rate=5e-7,
    variance_learning_rate=5e-7,
    cond_input_dim=6,
    hidden_dim=512,
    noise_dim=2,
    n_splits=5,
    n_epochs=5,
    batch_size=32,
    weight_decay=1e-6,
    optim_betas=(0.9, 0.99),
    device="cuda",
    output_folder="./../processed_data/chorus_neural_network/models/v1/trained_models/"
):
    kfold = GroupKFold(n_splits=n_splits, shuffle=True)

    trained_models = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(y, groups=g_data)):
        print(f"\nFold {fold + 1}/{n_splits}\n")

        y_train = y[train_idx]
        y_test = y[test_idx]

        train_dataset = CustomDataset(
            X_conditional[train_idx, :][y_train != 0, :], X_convolutional[train_idx, :][y_train != 0, :], y_train[y_train != 0]
        )
        test_dataset = CustomDataset(
            X_conditional[test_idx, :], X_convolutional[test_idx, :], y_test
        )

        y_train = y_train[y_train != 0]

        training_sampler_weights = get_sampler_weights(y_train)

        print("Creating Test Dataloader..")

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

        print("Creating Regressor..")

        bias_model = bias_model_class(input_dim=cond_input_dim, hidden_dim=hidden_dim, noise_dim=noise_dim)
        bias_model = bias_model.to(device)
        var_model = variance_model_class(input_dim=cond_input_dim, hidden_dim=hidden_dim)
        var_model = var_model.to(device)
        bias_optimizer = optim.Adam(bias_model.parameters(), lr=bias_learning_rate, weight_decay=weight_decay, betas=optim_betas)
        var_optimizer = optim.Adam(var_model.parameters(), lr=variance_learning_rate, weight_decay=weight_decay, betas=optim_betas)
        gaussian_nll = torch.nn.GaussianNLLLoss(eps=1e-9).to(device)

        print("Training...\n")

        train_model(
            bias_model=bias_model,
            var_model=var_model,
            bias_optimizer=bias_optimizer,
            var_optimizer=var_optimizer,
            gaussian_nll=gaussian_nll,
            training_sampler_weights=training_sampler_weights,
            training_dataset=train_dataset,
            batch_size=batch_size,
            testing_dataloader=test_loader,
            n_epochs=n_epochs,
            device=device,
        )

        trained_models.append({"bias" : bias_model, "variance" : var_model})

        torch.save(bias_model.state_dict(), os.path.join(os.path.abspath(output_folder), f"bias_model_{fold}.pth"))
        torch.save(var_model.state_dict(), os.path.join(os.path.abspath(output_folder), f"var_model_{fold}.pth"))
        np.savez(os.path.join(os.path.abspath(output_folder), f"train_test_sets_{fold}.npz"), train=train_idx, test=test_idx)

    return trained_models

# Training function
def train_model(
    bias_model,
    var_model,
    bias_optimizer,
    var_optimizer,
    gaussian_nll,
    training_sampler_weights,
    training_dataset,
    batch_size,
    testing_dataloader,
    n_epochs=10,
    device="cuda",
    beta=0.5
):

    torch.backends.cudnn.benchmark = True
    bias_model.train()
    var_model.train()
    for epoch in range(n_epochs):

        total_bias_loss = torch.as_tensor(0).type(torch.float32).to(device)

        training_sampler = torch.utils.data.WeightedRandomSampler(
            weights=training_sampler_weights,
            num_samples=len(training_dataset),
            replacement=True,  # Sample with replacement
        )

        # Create loaders
        train_loader = torch.utils.data.DataLoader(
            training_dataset, batch_size=batch_size, sampler=training_sampler, num_workers=2, pin_memory=True
        )

        for b, batch in enumerate(tqdm.tqdm(train_loader)):


            X_conditional, X_convolutional, y = batch
            X_conditional = X_conditional.to(device).type(torch.float32)
            X_convolutional = X_convolutional.to(device).type(torch.float32)
            y = y.to(device).type(torch.float32)


            bias_optimizer.zero_grad(set_to_none=True)
            pred_mean = bias_model(X_conditional, X_convolutional, torch.randn(size=(len(y), 2)).to(device))
            bias_loss = gaussian_nll(pred_mean, y, 10 * pred_mean)

            bias_loss.backward()
            bias_optimizer.step()

            with torch.no_grad():

                total_bias_loss += bias_loss

        print(f"\nEpoch : {epoch + 1} / {n_epochs}")
        print(f"Average Training bias_loss : {np.sqrt(total_bias_loss.item() / len(train_loader))}")

        if epoch % 10 == 9:
            
            avg_rmse_loss_nonzero, avg_var_loss_nonzero, avg_nll_loss_nonzero = evaluate_model(
                bias_model=bias_model, var_model=var_model, gaussian_nll=gaussian_nll, loader=testing_dataloader, device=device
            )
            print(f"Average Test nll_loss on non_zeros: {avg_nll_loss_nonzero}")
            print(f"Average Test rmse_loss on non_zeros: {avg_rmse_loss_nonzero}")
            print(f"Average Test var_loss on non_zeros: {avg_var_loss_nonzero}")
    
