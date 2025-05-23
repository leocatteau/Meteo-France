import torch
import json

import sys
sys.path.append('..')

########################################################################
from data_provider.data_provider import DataProvider
from models.GRIN import GRINet
from trainer.Filler import Filler

from types import SimpleNamespace


def main():
    data_kwargs = SimpleNamespace()
    data_kwargs.data = 'bdclim_clean'
    data_kwargs.dataset = 'WindowHorizonDataset'
    data_kwargs.root_path = '../../datasets/'
    data_kwargs.data_path = 'bdclim_safran_2023-2024.nc'
    data_kwargs.has_predictors = False
    data_kwargs.scaler = None
    data_kwargs.batch_size = 15
    data_kwargs.mask_length = 24*7*3
    data_kwargs.mask_proba = 0.5
    data_kwargs.window = 24*1*1
    data_kwargs.horizon = 0

    data_provider = DataProvider(data_kwargs)
    adjacency_matrix = torch.FloatTensor(data_provider.data.umap_adjacency(threshold=0.0, verbose=False)).to('cuda:0' if torch.cuda.is_available() else 'cpu')
    #adjacency_matrix = torch.FloatTensor(data_provider.data.correlation_adjacency(threshold=0.9, verbose=False)).to('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataloader = data_provider.train_dataloader()
    test_dataloader = data_provider.test_dataloader()

    model_kwargs = dict(adj=adjacency_matrix, d_in=1, d_ff=data_provider.data.n_nodes, global_att=True, d_hidden=256)
    filler_kwargs = SimpleNamespace()
    filler_kwargs.lr = 1e-4
    filler_kwargs.epochs = 10
    filler_kwargs.keep_proba = 1-data_kwargs.mask_proba

    filler = Filler(GRINet, model_kwargs, filler_kwargs)

    train_loss, test_loss = filler.train(train_dataloader=train_dataloader, test_dataloader=test_dataloader)
    filler.save_model('../trained_models/GRINet.pt')

    results = {
        'train_loss': train_loss,
        'test_loss': test_loss
    }
    with open('../../results/train_GRINet.json', 'w') as file:
        json.dump(results, file, indent=4)

    print("Training completed. Results saved to ../../results/train_GRINet.json")

if __name__ == "__main__":
    main()

