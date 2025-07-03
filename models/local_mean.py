import torch
import torch.nn as nn
from einops import rearrange
from sklearn.cluster import KMeans

from utils.functions import torch_nan_to_num


class local_mean(nn.Module):
    def __init__(self, adj, predictors, temporal=False):
        super(local_mean, self).__init__()
        self.model = nn.Identity()
        self.temporal = temporal
        self.adj = adj
        self.num_clusters = int(self.adj.shape[0]*0.5)
        self.predictors = predictors

        self.cluster_labels = torch.tensor(
            self.cluster_from_adjacency(self.adj, num_clusters=self.num_clusters),
            dtype=torch.long
        )

    def forward(self, x, mask, **kwargs):
        if self.temporal:
            x = rearrange(x, 'b s n c -> b n s c')
            mask = rearrange(mask, 'b s n c -> b n s c')

        x[~mask] = torch.nan

        imputations = []
        predictions = []
        for step in range(x.shape[1]):
            prediction = self.model(x[:,step,:,:].squeeze()).unsqueeze(-1)

            # # Normal ratio with geographical coordinates method
            coords = self.predictors[['lat', 'lon']].values
            coords = torch.tensor(coords, dtype=torch.float32, device=x.device)  # (num_stations, 2)

            # station_means = torch.nanmean(prediction, dim=1)  # shape: (1, num_stations)
            # # Compute inverse squared distance matrix (GC weighting)
            # print(coords.shape)
            # weights = (1/(coords[:,0]**2 + coords[:,1]**2)) / torch.sum(1/(coords[:,0]**2 + coords[:,1]**2), dim=0)
            # print(weights.shape)
            # print(station_means.shape)
            # # Weighted imputation
            # GC_coef =weights * station_means.co
            # mean_complete = weights * station_means
            # print('mean_complete', mean_complete.shape)

            # prediction = torch_nan_to_num(prediction, nan=0.0)

            # # Compute inverse squared distance matrix (GC weighting)
            # target_coords = coords.unsqueeze(0)  # (1, N, 2)
            # source_coords = coords.unsqueeze(1)  # (N, 1, 2)
            # squared_distances = torch.sum((target_coords - source_coords) ** 2, dim=-1)  # (N, N)
            # inv_distances = 1.0 / (squared_distances + 1e-8)
            # inv_distances.fill_diagonal_(0)

            # weights = inv_distances / inv_distances.sum(dim=1, keepdim=True)

            # # Weighted imputation
            # mean_complete = torch.matmul(weights, prediction.squeeze()).unsqueeze(-1)  # (N, 1)

            # labels = torch.tensor(self.cluster_from_adjacency(self.adj, num_clusters=self.num_clusters))
            labels = self.cluster_labels.to(x.device)
            means = self.scatter_mean(prediction.squeeze(), labels, num_clusters=self.num_clusters)
            mean_complete = means[labels].unsqueeze(-1)

            imputation = torch.where(mask[:,step,:,:], x[:,step,:,:], mean_complete)
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

    def cluster_from_adjacency(self, adj_matrix, num_clusters):
        features = adj_matrix  
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(features)
        return labels
    
    def scatter_mean(self, values, indices, num_clusters):
        device = values.device
        indices = indices.long()
        valid_mask = ~torch.isnan(values)
        values = values[valid_mask]
        indices = indices[valid_mask]
        sums = torch.zeros(num_clusters, device=device).scatter_add_(0, indices, values)
        counts = torch.zeros(num_clusters, device=device).scatter_add_(0, indices, torch.ones_like(values))
        means = sums / torch.clamp(counts, min=1)  # avoid div by zero
        return means