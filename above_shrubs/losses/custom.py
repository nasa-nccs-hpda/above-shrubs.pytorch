import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Custom loss function that penalizes predictions lower than targets
class PenalizedLowerPredictionLoss(nn.Module):
    def __init__(self, penalty_factor=2.0, use_mae=True):
        super(PenalizedLowerPredictionLoss, self).__init__()
        self.penalty_factor = penalty_factor
        self.base_loss = nn.L1Loss() if use_mae else nn.MSELoss()

    def forward(self, predictions, targets):
        base_loss = self.base_loss(predictions, targets)
        penalty_mask = predictions < targets
        penalty_loss = torch.mean(torch.abs(predictions[penalty_mask] - targets[penalty_mask]))
        total_loss = base_loss + self.penalty_factor * penalty_loss
        return total_loss

# Generate synthetic data for demonstration
torch.manual_seed(0)
X = torch.rand(100, 1) * 10  # Features
y = X * 2 + 3 + torch.randn(100, 1)  # Targets with some noise

# Dataset and DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Simple Linear Model
model = nn.Linear(1, 1)

# Loss function and optimizer
penalty_factor = 3.0
criterion = PenalizedLowerPredictionLoss(penalty_factor=penalty_factor, use_mae=True)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Final model parameters
print("Final model parameters:", list(model.parameters()))

