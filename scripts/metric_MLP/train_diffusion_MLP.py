import numpy as np
import torch
import matplotlib.pyplot as plt
import json

import sys
sys.path.append('../..')

########################################################################
from data_provider.data_provider import DataProvider
from models.MLP import MLP
from training.training import Trainer

from types import SimpleNamespace

def train_diffusion_step(masking_proba=0.5, first_pass=False, model_path='../../../results/metric_MLP/model/MLP_diffusion.pt'):
    data_kwargs = SimpleNamespace()
    data_kwargs.data = 'bdclim_clean'
    data_kwargs.dataset = 'WindowHorizonDataset'
    data_kwargs.root_path = '../../../datasets/'
    data_kwargs.data_path = 'bdclim_safran_2022-2024.nc'
    data_kwargs.has_predictors = False
    data_kwargs.scaler = None
    data_kwargs.batch_size = 15
    data_kwargs.mask_length = 24*3*1
    data_kwargs.mask_proba = masking_proba
    data_kwargs.window = 24*1*1
    data_kwargs.horizon = 0

    data_provider = DataProvider(data_kwargs)
    train_dataloader = data_provider.train_dataloader()
    test_dataloader = data_provider.test_dataloader()

    model_kwargs = dict(seq_dim=data_provider.data.n_nodes, hidden_dim=64)
    filler_kwargs = SimpleNamespace()
    filler_kwargs.lr = 5e-4
    filler_kwargs.epochs = 1
    filler_kwargs.keep_proba = 1-data_kwargs.mask_proba

    filler = Trainer(MLP, model_kwargs, filler_kwargs)
    if not first_pass:
        filler.load_model(model_path)
    train_loss, test_loss = filler.train(train_dataloader=train_dataloader, test_dataloader=test_dataloader)
    filler.save_model(model_path)

    return train_loss, test_loss



def main(epochs = 100):
    # masking_probas = np.linspace(0.1, 0.9, num=epochs//5).tolist()
    # masking_probas = np.random.beta(1.3, 2.5, size=epochs).tolist()
    masking_probas = np.random.uniform(0.05, 0.9, size=epochs).tolist()
    # masking_probas = [0.5]
    # masking_probas = np.array(1-np.sqrt(1-np.random.uniform(0.05, 0.95, size=epochs))).tolist() # sample from 1-x
    train_losses = []
    test_losses = []
    first_pass = True
    for masking_proba in masking_probas:
        train_loss, test_loss = train_diffusion_step(masking_proba=masking_proba, first_pass=first_pass)
        train_losses.extend(train_loss)
        test_losses.extend(test_loss)
        first_pass = False

    results = {
        'train_loss': train_losses,
        'test_loss': test_losses
    }
    with open('../../../results/metric_MLP/data/train_MLP_diffusion.json', 'w') as file:
        json.dump(results, file, indent=4)

if __name__ == "__main__":
    epochs = 100 
    main(epochs=epochs)

