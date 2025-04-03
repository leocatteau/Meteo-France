import torch
import torch.nn as nn

from models.baseline import mean_fill
from utils.functions import torch_nan_to_num


class MLP(nn.Module):
    def __init__(self, seq_dim, hidden_dim):
        super(MLP, self).__init__()
        # self.model = nn.ModuleList([nn.Linear(seq_dim, hidden_dims[0]), nn.ReLU()])
        # for i in range(1, len(hidden_dims)):
        #     self.model.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        #     self.model.append(nn.ReLU())
        # self.model.append(nn.Linear(hidden_dims[-1], seq_dim))

        self.model = nn.Sequential(
            nn.Linear(seq_dim, int(0.5*hidden_dim)),
            nn.Sigmoid(),
            #nn.BatchNorm1d(int(0.5*hidden_dim)),
            nn.Linear(int(0.5*hidden_dim), hidden_dim),
            nn.Sigmoid(),
            #nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, int(0.5*hidden_dim)),
            nn.Sigmoid(),
            #nn.BatchNorm1d(int(0.5*hidden_dim)),
            nn.Linear(int(0.5*hidden_dim), seq_dim)
        )

    def forward(self, x):
        mask = torch.isnan(x)
        mean_model = mean_fill(columnwise=True)
        x_mean = mean_model(x)
        y_pred = self.model(x_mean)
        y_pred = y_pred*(mask) + torch_nan_to_num(x)
        return y_pred