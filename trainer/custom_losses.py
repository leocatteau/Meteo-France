import torch 


def masked_MSE(y_true, y_pred, mask):
    # should the model be trained only to fitting holes? this reduces a lot the data available to learn the process
    mse = torch.mean((y_true[~mask] - y_pred[~mask]) ** 2)
    return mse

def spatiotemporal_masked_MSE(y_true, y_pred, mask, spatial_weight=0.5):
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    mask = mask.squeeze()
    spatial_weight = (y_true.shape[1]/y_true.shape[2]) * spatial_weight # normalize by the different sizes in spatial and temporal dimensions
    spatial_mse = torch.mean((y_true*(~mask) - y_pred*(~mask)) ** 2, dim=2)
    temporal_mse = torch.mean((y_true*(~mask) - y_pred*(~mask)) ** 2, dim=1)
    spatiotemporal_mse = torch.mean(spatial_weight * spatial_mse) + torch.mean((1 - spatial_weight) * temporal_mse)
    return spatiotemporal_mse