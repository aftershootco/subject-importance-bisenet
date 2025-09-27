import torch.nn as nn
import torch
import torch.nn.functional as F

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-5

    def forward(self, output, target):
        assert output.size() == target.size(), "'input' and 'target' must have the same shape"
        output = F.softmax(output, dim=1)
        output = flatten(output)
        target = flatten(target)
        # intersect = (output * target).sum(-1).sum() + self.epsilon
        # denominator = ((output + target).sum(-1)).sum() + self.epsilon

        intersect = (output * target).sum(-1)
        denominator = (output + target).sum(-1)
        dice = intersect / denominator
        dice = torch.mean(dice)
        return 1 - dice
        # return 1 - 2. * intersect / denominator


class BinaryDiceLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, output, target):
        # output: [N, 1, H, W], raw logits
        # target: [N, H, W], 0 or 1
        output = torch.sigmoid(output)   # convert logits → probs
        output = output.view(-1)
        target = target.view(-1).float()

        intersect = (output * target).sum()
        denominator = output.sum() + target.sum() + self.eps
        dice = (2 * intersect + self.eps) / denominator
        return 1 - dice

class BCEWithDiceLoss(nn.Module):
    def __init__(self, w_dice=1.0, w_bce=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = BinaryDiceLoss()
        self.w_dice = w_dice
        self.w_bce = w_bce

    def forward(self, output, target):
        return self.w_bce * self.bce(output, target.float()) + \
               self.w_dice * self.dice(output, target)