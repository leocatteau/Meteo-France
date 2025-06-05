import torch
import json

import sys
sys.path.append('../..')

########################################################################
from data_provider.data_provider import DataProvider
from models.GRIN import GRINet
from trainer.Trainer import Trainer

from types import SimpleNamespace


def main(hidden_dim=100):
    data_kwargs = SimpleNamespace()
    data_kwargs.data = 'bdclim_clean'
    data_kwargs.dataset = 'WindowHorizonDataset'
    data_kwargs.root_path = '../../../datasets/'
    data_kwargs.data_path = 'bdclim_safran_2023-2024.nc'
    data_kwargs.has_predictors = False
    data_kwargs.scaler = None
    data_kwargs.batch_size = 15
    print(data_kwargs.batch_size)
    data_kwargs.mask_length = 24*7*3
    data_kwargs.mask_proba = 0.5
    data_kwargs.window = 6*1*1
    data_kwargs.horizon = 0

    data_provider = DataProvider(data_kwargs)
    #adjacency_matrix, graph = data_provider.data.correlation_adjacency(threshold=0.9, verbose=False)
    adjacency_matrix, graph = data_provider.data.KNN_adjacency(threshold=0.0, verbose=False)
    print("adjacency matrix created")
    #adjacency_matrix, graph = data_provider.data.umap_adjacency(threshold=0.0, verbose=False)
    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
    train_dataloader = data_provider.train_dataloader()
    test_dataloader = data_provider.test_dataloader()

    model_kwargs = dict(adj=adjacency_matrix, d_in=1, d_ff=hidden_dim, global_att=True, d_hidden_spatial=hidden_dim, d_hidden_temporal=data_kwargs.window)
    filler_kwargs = SimpleNamespace()
    filler_kwargs.lr = 5e-4
    filler_kwargs.epochs = 3
    filler_kwargs.keep_proba = 1-data_kwargs.mask_proba

    filler = Trainer(GRINet, model_kwargs, filler_kwargs)

    print('model created with {} hidden dim'.format(hidden_dim))

    train_loss, test_loss = filler.train(train_dataloader=train_dataloader, test_dataloader=test_dataloader)
    filler.save_model('../../../results/hyperparam_GRIN/GRINet_hiddendim_{}.pt'.format(hidden_dim))

    results = {
        'train_loss': train_loss,
        'test_loss': test_loss
    }
    with open('../../../results/hyperparam_GRIN/train_GRINet_hiddendim_{}.json'.format(hidden_dim), 'w') as file:
        json.dump(results, file, indent=4)

if __name__ == "__main__":
    hidden_dim = int(sys.argv[1])
    print('in main')
    main(hidden_dim=hidden_dim)
