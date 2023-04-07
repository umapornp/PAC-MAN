import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import log_softmax, cross_entropy


class CrossEntropyHungarianLoss(nn.Module):
    """ Cross Entropy loss function. """

    def __init__(self, ignore_index=-1, reduction='mean'):
        """ Constructor for `CrossEntropyHungarianLoss`.
        
        Args:
            ignore_index: Ignore index.
            reduction: Reduction.
        """
        super(CrossEntropyHungarianLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction


    def forward(self, logits, labels):
        """ Process of `CrossEntropyHungarianLoss`.
        
        Args:
            logits: Logits.
            labels: Labels.

        Returns:
        - loss : Loss.
        """
        # Rearrange optimal labels using the hungarian algorithm.
        new_labels = self._hungarian_matcher(logits, labels)

        # Compute loss.
        num_classes = logits.shape[-1]
        loss = cross_entropy(logits.contiguous().view(-1, num_classes),
                            new_labels.view(-1),
                            ignore_index=self.ignore_index,
                            reduction=self.reduction)
        return loss


    @torch.no_grad()
    def _hungarian_matcher(self, logits, labels):
        """ Hungarian algorithm. 
        
        Args:
            logits: Logits.
            labels: Labels.
        
        Returns:
            labels: Optimal labels that provide the minimum loss.
        """
        batch_size = logits.shape[0]
        log_prob = -log_softmax(logits, dim=-1) # Cross entropy loss.

        for bs in range(batch_size):
            # Construct a cost tensor.
            labels_mask = (labels[bs] > -1).nonzero(as_tuple=True)
            cost_tensor = log_prob[bs][labels_mask][:, labels[bs][labels_mask]].to(torch.float16)
            
            # Perform the hungarian algorithm.
            _, col = linear_sum_assignment(cost_tensor)

            # Rearrange labels.
            labels[bs][labels_mask] = labels[bs][labels_mask][col]

        return labels