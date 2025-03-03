import torch
import torch.nn as nn

class PenalizedLowerPredictionLoss(nn.Module):
    def __init__(self, penalty_factor=2.0, use_mae=True):
        """
        Custom loss that penalizes predictions that are lower than the target.

        Parameters:
        - penalty_factor (float): Factor to increase the penalty for lower-than-target predictions.
        - use_mae (bool): If True, use MAE as the base loss; if False, use MSE.
        """
        super(PenalizedLowerPredictionLoss, self).__init__()
        self.penalty_factor = penalty_factor
        self.base_loss = nn.L1Loss() if use_mae else nn.MSELoss()

    def forward(self, predictions, targets):
        # Calculate standard base loss (either MAE or MSE)
        base_loss = self.base_loss(predictions, targets)

        # Find where predictions are lower than the targets
        penalty_mask = predictions < targets
        
        # Calculate additional penalty for predictions lower than the target
        penalty_loss = torch.mean(torch.abs(predictions[penalty_mask] - targets[penalty_mask]))
        
        # Combine base loss with penalized term, scaled by penalty factor
        total_loss = base_loss + self.penalty_factor * penalty_loss
        
        return total_loss

# Initialize the custom penalized loss
penalty_factor = 3.0  # Example penalty factor for low predictions
loss_fn = PenalizedLowerPredictionLoss(penalty_factor=penalty_factor, use_mae=True)

# Example predictions and targets
predictions = torch.tensor([7.0, 8.5, 9.0, 5.0])  # Example predictions
targets = torch.tensor([8.0, 8.0, 8.0, 8.0])       # Example targets

loss = loss_fn(predictions, targets)
print("Custom Penalized Lower Prediction Loss:", loss.item())

