import json

import sys
sys.path.append('..')

########################################################################
from data_provider.data_provider import DataProvider
from models.linear_MLP import linear_MLP
from trainer.Filler import Filler

from types import SimpleNamespace


def main(verbose=False):
    data_kwargs = SimpleNamespace()
    data_kwargs.data = 'bdclim_clean'
    data_kwargs.dataset = 'SequenceMaskDataset'
    data_kwargs.root_path = '../../datasets/'
    data_kwargs.data_path = 'bdclim_safran_2023-2024.nc'
    data_kwargs.has_predictors = False
    data_kwargs.scaler = None
    data_kwargs.batch_size = 1
    data_kwargs.mask_length = 24*7*3
    data_kwargs.mask_proba = 0.5

    data_provider = DataProvider(data_kwargs)
    train_dataloader = data_provider.train_dataloader()
    test_dataloader = data_provider.test_dataloader()

    model_kwargs = dict(seq_dim=data_provider.data.n_nodes)
    filler_kwargs = SimpleNamespace()
    filler_kwargs.lr = 1e-5
    filler_kwargs.epochs = 100
    filler_kwargs.keep_proba = 1-data_kwargs.mask_proba

    filler = Filler(linear_MLP, model_kwargs, filler_kwargs)

    train_loss, test_loss = filler.train(train_dataloader=train_dataloader, test_dataloader=test_dataloader)
    filler.save_model('../trained_models/linear_MLP.pt')

    results = {
        'train_loss': train_loss,
        'test_loss': test_loss
    }
    with open('../../results/linear_MLP.json', 'w') as file:
        json.dump(results, file, indent=4)

if __name__ == "__main__":
    main(verbose=True)



