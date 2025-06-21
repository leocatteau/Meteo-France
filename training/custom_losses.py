import torch 
import numpy as np
import networkx as nx

from sklearn.cluster import KMeans

def masked_MSE(y_true, y_pred, mask):
    # should the model be trained only to fitting holes? this reduces a lot the data available to learn the process
    mse = torch.mean((y_true[~mask] - y_pred[~mask]) ** 2)
    return mse

def masked_RMSE(y_true, y_pred, mask):
    # should the model be trained only to fitting holes? this reduces a lot the data available to learn the process
    rmse = torch.sqrt(torch.mean((y_true[~mask] - y_pred[~mask]) ** 2))
    print(mask.sum(), mask.numel())
    print(f'Masked MSE: {rmse.item()}')
    return rmse

def masked_MAE(y_true, y_pred, mask):
    # should the model be trained only to fitting holes? this reduces a lot the data available to learn the process
    mae = torch.mean(torch.abs(y_true[~mask] - y_pred[~mask]))
    return mae

def spatiotemporal_masked_MSE(y_true, y_pred, mask, spatial_weight=0.5):
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    mask = mask.squeeze()
    spatial_weight = (y_true.shape[1]/y_true.shape[2]) * spatial_weight # normalize by the different sizes in spatial and temporal dimensions
    spatial_mse = torch.mean((y_true*(~mask) - y_pred*(~mask)) ** 2, dim=2)
    temporal_mse = torch.mean((y_true*(~mask) - y_pred*(~mask)) ** 2, dim=1)
    spatiotemporal_mse = torch.mean(spatial_weight * spatial_mse) + torch.mean((1 - spatial_weight) * temporal_mse)
    return spatiotemporal_mse

def temporal_gradient_MSE(y_true, y_pred, mask):
    grad_y_true = torch.gradient(y_true, dim=1)[0]
    grad_y_pred = torch.gradient(y_pred, dim=1)[0]
    grad_mse = torch.mean((grad_y_true[~mask] - grad_y_pred[~mask]) ** 2)
    # grad_mse = torch.mean(grad_y_pred[~mask] ** 2) # if we want to minimize gradients
    return grad_mse

def spatial_graph_gradient_MSE(y_true, y_pred, mask, graph):
    loss = 0 
    for i, j in graph.edges():
        diff_pred = y_pred[:,:,i,:] - y_pred[:,:,j,:]
        diff_true = y_true[:,:,i,:] - y_true[:,:,j,:]
        #loss += torch.mean((diff_pred[~mask] - diff_true[~mask[:,:,j]]) ** 2) # it is hard to enforce the calculation only on masked because it supposes double masking
        loss += torch.mean((diff_pred - diff_true) ** 2) 
        # loss += torch.mean(diff_pred[~mask] ** 2) # if we want to minimize gradients
    return loss / len(graph.edges())

def spatial_laplacian_MSE(y_true, y_pred, mask, graph):
    L = torch.tensor(nx.laplacian_matrix(graph).todense(), dtype=torch.float32, requires_grad=True)
    y_pred = y_pred.view(-1, y_pred.shape[2])
    loss = torch.trace(torch.matmul(y_pred, torch.matmul(L, y_pred.T)))/(y_pred.shape[0] * y_pred.shape[1])
    return loss

def cluster_from_adjacency(adj_matrix, num_clusters):
    features = adj_matrix  
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(features)
    return labels

def scatter_mean(values, indices, num_clusters):
    """
    Pure PyTorch replacement for torch_scatter.scatter_mean
    values: Tensor [N_total]
    indices: LongTensor [N_total] with cluster IDs in [0, num_clusters)
    num_clusters: int
    Returns: mean value for each cluster [num_clusters]
    """
    device = values.device
    indices = indices.long()
    valid_mask = ~torch.isnan(values)
    values = values[valid_mask]
    indices = indices[valid_mask]
    sums = torch.zeros(num_clusters, device=device).scatter_add_(0, indices, values)
    counts = torch.zeros(num_clusters, device=device).scatter_add_(0, indices, torch.ones_like(values))
    means = sums / torch.clamp(counts, min=1)  # avoid div by zero
    return means

def coarse_rmse(pred, target, cluster_labels):
    B, S, N, _ = pred.shape
    pred_flat = pred.view(-1, N)  # [B*S, N]
    target_flat = target.view(-1, N)

    cluster_labels = torch.tensor(cluster_labels, device=pred.device)
    cluster_labels = cluster_labels.unsqueeze(0).expand(B * S, -1).reshape(-1)

    pred_vals = pred_flat.reshape(-1)
    target_vals = target_flat.reshape(-1)

    num_clusters = cluster_labels.max().item() + 1
    pred_mean = scatter_mean(pred_vals, cluster_labels, num_clusters)
    target_mean = scatter_mean(target_vals, cluster_labels, num_clusters)

    return torch.sqrt(torch.mean((pred_mean - target_mean) ** 2))

def coarse_mae(pred, target, cluster_labels):
    B, S, N, _ = pred.shape
    pred_flat = pred.view(-1, N)  # [B*S, N]
    target_flat = target.view(-1, N)

    cluster_labels = torch.tensor(cluster_labels, device=pred.device)
    cluster_labels = cluster_labels.unsqueeze(0).expand(B * S, -1).reshape(-1)

    pred_vals = pred_flat.reshape(-1)
    target_vals = target_flat.reshape(-1)

    num_clusters = cluster_labels.max().item() + 1
    pred_mean = scatter_mean(pred_vals, cluster_labels, num_clusters)
    target_mean = scatter_mean(target_vals, cluster_labels, num_clusters)

    return torch.mean(torch.abs(pred_mean - target_mean))

def RG_loss(pred, target, adjacency_matrix):
    N = adjacency_matrix.shape[0]
    scales = np.linspace(1, N // 2.5, num=5, dtype=int).tolist()
    weights = [1.0 / len(scales)] * len(scales)

    RMSE = 0.0
    MAE = 0.0
    for scale, w in zip(scales, weights):
        num_clusters = max(2, N // scale)
        labels = cluster_from_adjacency(adjacency_matrix, num_clusters)
        RMSE += w * coarse_rmse(pred, target, labels)
        MAE += w * coarse_mae(pred, target, labels)
    return RMSE, MAE

def masked_RG_RMSE_MAE(pred, target, adjacency_matrix, mask):
    pred = pred.unsqueeze(0) # [1, S, N, C]
    target = target.unsqueeze(0) # [1, S, N, C]
    mask = mask.unsqueeze(0) # [1, S, N, C]
    pred[~mask] = np.nan
    RMSE, MAE = RG_loss(pred, target, adjacency_matrix)
    return RMSE, MAE


def mixed_loss(y_true, y_pred, mask, spatial_weight=0.5, eta=0.1, graph=None):
    spatiotemporal_mse = spatiotemporal_masked_MSE(y_true, y_pred, mask, spatial_weight)
    # grad_mse = temporal_gradient_MSE(y_true, y_pred, mask)
    laplacian_mse = spatial_laplacian_MSE(y_true, y_pred, mask, graph=graph)
    return spatiotemporal_mse + eta * laplacian_mse