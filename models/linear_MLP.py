import torch
import torch.nn as nn


class linear_MLP(nn.Module):
    def __init__(self, seq_dim):
        super(linear_MLP, self).__init__()
        self.model = nn.Sequential(nn.Linear(seq_dim, seq_dim))

    def forward(self, x, mask, **kwargs):
        if len(x.shape) == 4:
            # [b s n c] -> [b*s n] #need to check if it changes order for sequence testing (doesn't really matter because nn doesn't have temporal correlations)
            x = x.view(-1, x.shape[2],x.shape[3])
            mask = mask.view(-1, mask.shape[2],mask.shape[3])
        x = x.squeeze()
        mask = mask.squeeze()

        prediction = self.model(x)
        imputation = prediction*(mask) + torch.nan_to_num(x)
        imputation = imputation.unsqueeze(-1).unsqueeze(1)
        prediction = prediction.unsqueeze(-1).unsqueeze(1)

        if self.training:
            return imputation, prediction
        return imputation
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()