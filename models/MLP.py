import torch
import torch.nn as nn

from utils.functions import torch_nan_to_num


# class MLP(nn.Module):
#     def __init__(self, seq_dim, hidden_dim):
#         super(MLP, self).__init__()
#         # self.model = nn.ModuleList([nn.Linear(seq_dim, hidden_dims[0]), nn.ReLU()])
#         # for i in range(1, len(hidden_dims)):
#         #     self.model.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
#         #     self.model.append(nn.ReLU())
#         # self.model.append(nn.Linear(hidden_dims[-1], seq_dim))

#         self.model = nn.Sequential(
#             nn.Linear(seq_dim, int(0.5*hidden_dim)),
#             nn.ReLU(),
#             #nn.BatchNorm1d(int(0.5*hidden_dim)),
#             nn.Linear(int(0.5*hidden_dim), hidden_dim),
#             nn.ReLU(),
#             #nn.BatchNorm1d(hidden_dim),
#             nn.Linear(hidden_dim, int(0.5*hidden_dim)),
#             nn.ReLU(),
#             #nn.BatchNorm1d(int(0.5*hidden_dim)),
#             nn.Linear(int(0.5*hidden_dim), seq_dim)
#         )

#     def forward(self, x, mask):
#         if len(x.shape) == 4:
#             # [b s n c] -> [b*s n] #need to check if it changes order for sequence testing (doesn't really matter because nn doesn't have temporal correlations)
#             x = x.squeeze(-1).view(-1, x.shape[2])
#             mask = mask.squeeze(-1).view(-1, x.shape[2])

#         prediction = self.model(x)
#         imputation = prediction*(mask) + torch_nan_to_num(x)

#         if self.training:
#             return imputation, prediction
#         return imputation
    
#     def save_model(self, path):
#         torch.save(self.state_dict(), path)

#     def load_model(self, path):
#         self.load_state_dict(torch.load(path))
#         self.eval()

class MLP(nn.Module):
    def __init__(self, seq_dim, hidden_dim):
        super(MLP, self).__init__()
        # self.model = nn.ModuleList([nn.Linear(seq_dim, hidden_dims[0]), nn.ReLU()])
        # for i in range(1, len(hidden_dims)):
        #     self.model.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        #     self.model.append(nn.ReLU())
        # self.model.append(nn.Linear(hidden_dims[-1], seq_dim))

        self.model = nn.Sequential(
            nn.Linear(seq_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), seq_dim)
        )

    def forward(self, x, mask, **kwargs):
        imputations = []
        predictions = []
        for step in range(x.shape[1]):
            prediction = self.model(x[:,step,:,:].squeeze()).unsqueeze(-1)
            imputation = prediction*(~mask[:,step,:,:]) + torch.nan_to_num(x[:,step,:,:])
            predictions.append(prediction)
            imputations.append(imputation)

        imputations = torch.stack(imputations, dim=1)
        predictions = torch.stack(predictions, dim=1)

        if self.training:
            return imputations, predictions
        return imputations 
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()