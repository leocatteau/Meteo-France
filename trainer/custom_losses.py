import torch 
import numpy as np
import networkx as nx

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
    L = torch.tensor(nx.laplacian_matrix(graph).todense())
    y_pred = y_pred.view(-1, y_pred.shape[2])
    loss = torch.trace(torch.matmul(y_pred.T, torch.matmul(L, y_pred)))/(y_pred.shape[0] * y_pred.shape[1])
    return loss

def RG_loss(y_true, y_pred, mask):
    pass

def mixed_loss(y_true, y_pred, mask, spatial_weight=0.5, eta=0.1):
    spatiotemporal_mse = spatiotemporal_masked_MSE(y_true, y_pred, mask, spatial_weight)
    grad_mse = temporal_gradient_MSE(y_true, y_pred, mask)
    return spatiotemporal_mse + eta * grad_mse