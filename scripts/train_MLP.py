import numpy as np
import torch
import matplotlib.pyplot as plt
import json

import sys
sys.path.append('..')

########################################################################
from data_provider.data_provider import DataProvider
from models.MLP import MLP
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
    train_dataloader = data_provider.train_dataloader()
    test_dataloader = data_provider.test_dataloader()

    model_kwargs = dict(seq_dim=data_provider.data.n_nodes, hidden_dim=2*data_provider.data.n_nodes)
    filler_kwargs = SimpleNamespace()
    filler_kwargs.lr = 1e-5
    filler_kwargs.epochs = 200
    filler_kwargs.keep_proba = 1-data_kwargs.mask_proba

    filler = Filler(MLP, model_kwargs, filler_kwargs)

    train_loss, test_loss = filler.train(train_dataloader=train_dataloader, test_dataloader=test_dataloader)
    filler.save_model('../trained_models/MLP.pt')

    results = {
        'train_loss': train_loss,
        'test_loss': test_loss
    }
    with open('../../results/train_MLP.json', 'w') as file:
        json.dump(results, file, indent=4)

if __name__ == "__main__":
    main()



