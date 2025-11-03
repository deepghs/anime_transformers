"""
This module implements the Focal Loss function for addressing class imbalance in classification tasks.

Focal Loss is particularly useful in scenarios where there is a significant imbalance between classes,
such as object detection where the number of background pixels far exceeds the number of foreground pixels.
The loss function down-weights easy examples and focuses training on hard negatives.
"""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss for multi-class classification tasks.

    Focal Loss is designed to address class imbalance by down-weighting easy examples
    and focusing learning on hard negatives. It modifies the standard cross-entropy loss
    by adding a modulating factor (1-p_t)^gamma that reduces the loss contribution from
    well-classified examples.

    Based on https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8

    The focal loss is defined as:
    FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)

    where p_t is the model's estimated probability for the true class.
    """

    def __init__(self, num_classes: int, gamma: float = 2., reduction: str = 'mean',
                 weight: Optional[torch.Tensor] = None):
        """
        Initialize the Focal Loss module.

        :param num_classes: Number of classes in the classification task.
        :type num_classes: int
        :param gamma: Focusing parameter. Higher gamma reduces the relative loss for well-classified examples.
                     When gamma=0, focal loss is equivalent to cross-entropy loss.
        :type gamma: float
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                         'none': no reduction will be applied,
                         'mean': the sum of the output will be divided by the number of elements,
                         'sum': the output will be summed.
        :type reduction: str
        :param weight: Manual rescaling weight given to each class. If given, has to be a Tensor of size num_classes.
        :type weight: Optional[torch.Tensor]
        """
        nn.Module.__init__(self)
        self.num_classes = num_classes
        weight = torch.as_tensor(weight).float() if weight is not None else weight
        self.register_buffer('weight', weight)
        self.weight: torch.Tensor

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the focal loss between logits and labels.

        :param logits: Raw predictions from the model of shape (N, C) where N is batch size and C is number of classes.
        :type logits: torch.Tensor
        :param labels: Ground truth class indices of shape (N,) with values in range [0, num_classes-1].
        :type labels: torch.Tensor
        :return: Computed focal loss. Shape depends on reduction parameter:
                - 'none': (N,) - loss for each sample
                - 'mean': scalar - mean loss across all samples  
                - 'sum': scalar - sum of losses across all samples
        :rtype: torch.Tensor
        """
        log_prob = F.log_softmax(logits, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            labels,
            weight=self.weight,
            reduction=self.reduction
        )


if __name__ == '__main__':
    logits = torch.randn(3, 5)
    labels = torch.randint(0, 4, (3,))
    print(logits, labels)

    fn_loss = FocalLoss(num_classes=5)
    print(fn_loss(logits, labels))

    fn_loss = FocalLoss(num_classes=5, reduction='none')
    print(fn_loss(logits, labels))
