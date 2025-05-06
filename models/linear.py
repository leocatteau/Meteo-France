import torch
import torch.nn as nn


# class linear_MLP(nn.Module):
#     def __init__(self, seq_dim):
#         super(linear_MLP, self).__init__()
#         self.model = nn.Sequential(nn.Linear(seq_dim, seq_dim))

#     def forward(self, x, mask, **kwargs):
#         if len(x.shape) == 4:
#             # [b s n c] -> [b*s n] #need to check if it changes order for sequence testing (doesn't really matter because nn doesn't have temporal correlations)
#             x = x.view(-1, x.shape[2],x.shape[3])
#             mask = mask.view(-1, mask.shape[2],mask.shape[3])
#         x = x.squeeze()
#         mask = mask.squeeze()

#         prediction = self.model(x)
#         imputation = prediction*(~mask) + torch.nan_to_num(x)
#         imputation = imputation.unsqueeze(-1).unsqueeze(1)
#         prediction = prediction.unsqueeze(-1).unsqueeze(1)

#         if self.training:
#             return imputation, prediction
#         return imputation
    
#     def save_model(self, path):
#         torch.save(self.state_dict(), path)

#     def load_model(self, path):
#         self.load_state_dict(torch.load(path))
#         self.eval()


# attempt for managing sequence as batch 
# class linear_MLP(nn.Module):
#     def __init__(self, seq_dim):
#         super(linear_MLP, self).__init__()
#         self.model = nn.Sequential(nn.Linear(seq_dim, seq_dim))

#     def forward(self, x, mask, **kwargs):
#         imputation = []
#         prediction = []
#         for batch, batch_mask in zip(x, mask):
#             batch = batch.squeeze()
#             batch_mask = batch_mask.squeeze()
#             batch_prediction = self.model(batch)
#             batch_imputation = batch_prediction*(~batch_mask) + torch.nan_to_num(batch)
#             batch_imputation = batch_imputation.unsqueeze(-1)
#             batch_prediction = batch_prediction.unsqueeze(-1)

#             prediction.append(batch_prediction)
#             imputation.append(batch_imputation)
#         imputation = torch.stack(imputation, dim=0)
#         prediction = torch.stack(prediction, dim=0)
        
#         if self.training:
#             return imputation, prediction
#         return imputation
    
#     def save_model(self, path):
#         torch.save(self.state_dict(), path)

#     def load_model(self, path):
#         self.load_state_dict(torch.load(path))
#         self.eval()


class linear(nn.Module):
    def __init__(self, seq_dim):
        super(linear, self).__init__()
        self.model = nn.Sequential(nn.Linear(seq_dim, seq_dim))

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