import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Focal Loss implementation.
        Args:
            alpha (float): Balance the importance of positive and negative samples. For class 1 (usually foreground or rare class), weight is alpha.
            gamma (float): Focusing parameter to reduce the loss contribution from easy-to-classify samples.
            reduction (str): 'none', 'mean', 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6  # Avoid log(0)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()  # Ensure targets are float type
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)  # Get the probability corresponding to the true label
        modulating_factor = torch.pow(torch.clamp(1.0 - p_t, min=self.eps), self.gamma)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss_elements = alpha_factor * modulating_factor * BCE_loss

        if self.reduction == 'mean':
            if focal_loss_elements.numel() == 0:
                return torch.tensor(0.0, device=inputs.device, requires_grad=inputs.requires_grad)
            return focal_loss_elements.mean()
        elif self.reduction == 'sum':
            return focal_loss_elements.sum()
        elif self.reduction == 'none':
            return focal_loss_elements
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")
