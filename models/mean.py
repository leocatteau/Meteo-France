import torch
import torch.nn as nn
from einops import rearrange

from utils.functions import torch_nan_to_num


class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()
        self.model = nn.Identity()

    def forward(self, x, mask, **kwargs):

        imputations = []
        predictions = []
        for step in range(x.shape[1]):
            prediction = self.model(x[:,step,:,:].squeeze()).unsqueeze(-1)
            # imputation = prediction*(~mask[:,step,:,:]) + torch.nan_to_num(x[:,step,:,:])
            imputation = torch.where(mask[:,step,:,:], x[:,step,:,:], prediction)
            predictions.append(prediction)
            imputations.append(imputation)

        imputations = torch.stack(imputations, dim=1)
        predictions = torch.stack(predictions, dim=1)

        if self.temporal:
            imputations = rearrange(imputations, 'b n s c -> b s n c')
            predictions = rearrange(predictions, 'b n s c -> b s n c')

        if self.training:
            return imputations, predictions
        return imputations 
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()