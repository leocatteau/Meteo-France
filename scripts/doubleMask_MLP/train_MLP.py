import numpy as np
import torch
import matplotlib.pyplot as plt
import json

import sys
sys.path.append('../..')  # Adjust the path to include the parent directory

########################################################################
from data_provider.data_provider import DataProvider
from models.MLP import MLP
from trainer.Trainer import Trainer

from types import SimpleNamespace


def main():
    data_kwargs = SimpleNamespace()
    data_kwargs.data = 'bdclim'
    data_kwargs.dataset = 'WindowHorizonDataset'
    data_kwargs.root_path = '../../../datasets'
    data_kwargs.data_path = 'bdclim_safran_2020-2024.nc'
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

    model_kwargs = dict(seq_dim=data_provider.data.n_nodes, hidden_dim=512)
    filler_kwargs = SimpleNamespace()
    filler_kwargs.lr = 5e-4
    filler_kwargs.epochs = 100
    filler_kwargs.keep_proba = 1-data_kwargs.mask_proba

    filler = Trainer(MLP, model_kwargs, filler_kwargs)

    train_loss, test_loss = filler.train(train_dataloader=train_dataloader, test_dataloader=test_dataloader)
    filler.save_model('../../../results/DoubleMask_MLP/model/MLP_100.pt')

    results = {
        'train_loss': train_loss,
        'test_loss': test_loss
    }
    with open('../../../results/DoubleMask_MLP/data/train_MLP_100.json', 'w') as file:
        json.dump(results, file, indent=4)

if __name__ == "__main__":
    main()

