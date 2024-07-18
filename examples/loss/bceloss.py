import torch.nn as nn
import torch.nn.functional as F

class BCELoss(nn.Module):
    '''Binary Cross Entroy Loss for GW detection'''
    def __init__(self):
        super(BCELoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, prediction, target):
        target = target if len(target.shape) == 1 else target.squeeze(1)
        return self.criterion(prediction, target)

# import torch.nn as nn
# import torch.nn.functional as F

# class BCELoss(nn.Module):
#     '''Binary Cross Entroy Loss for GW detection'''
#     def __init__(self):
#         super(BCELoss, self).__init__()
#         self.criterion = F.binary_cross_entropy()

#     def forward(self, prediction, target):
#         target = target if len(target.shape) == 1 else target.squeeze(1)
#         return self.criterion(prediction, target)