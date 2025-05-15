import torch
import json

import sys
sys.path.append('..')

########################################################################
from data_provider.data_provider import DataProvider
from models.baseline import mean_fill, svd, random_forest, linear

from types import SimpleNamespace


def main():
    data_kwargs = SimpleNamespace()
    data_kwargs.data = 'bdclim_clean'
    data_kwargs.dataset = 'WindowHorizonDataset'
    data_kwargs.root_path = '../../datasets/'
    data_kwargs.data_path = 'bdclim_safran_2023-2024.nc'
    data_kwargs.has_predictors = False
    data_kwargs.scaler = None
    data_kwargs.batch_size = 1
    data_kwargs.mask_length = 24*7*3
    data_kwargs.mask_proba = 0.5
    data_kwargs.window = 24*7*1
    data_kwargs.horizon = 0

    data_provider = DataProvider(data_kwargs)
    clean_data = data_provider.dataset.data
    corrupted_data = data_provider.dataset.corrupted_data
    eval_mask = data_provider.dataset.eval_mask

    X_train = corrupted_data[:int(corrupted_data.shape[0] * 0.5)]
    y_train = clean_data[:int(clean_data.shape[0] * 0.5)]

    mean_model = mean_fill()
    reconstructed_data_mean = mean_model(torch.FloatTensor(corrupted_data))

    svd_model = svd()
    svd_model.train(torch.FloatTensor(X_train), torch.FloatTensor(y_train), verbose=True)
    reconstructed_data_svd = svd_model(torch.FloatTensor(corrupted_data))

    random_forest_model = random_forest()
    random_forest_model.train(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    reconstructed_data_rf = random_forest_model(torch.FloatTensor(corrupted_data))

    linear_model = linear()
    linear_model.train(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    reconstructed_data_linear = linear_model(torch.FloatTensor(corrupted_data))

    results = {
        'clean_data': clean_data.detach().cpu().numpy().tolist(),
        'eval_mask': eval_mask.detach().cpu().numpy().tolist(),
        'reconstructed_data_mean': reconstructed_data_mean.detach().cpu().numpy().tolist(),
        'reconstructed_data_svd': reconstructed_data_svd.detach().cpu().numpy().tolist(),
        'reconstructed_data_rf': reconstructed_data_rf.detach().cpu().numpy().tolist(),
        'reconstructed_data_linear': reconstructed_data_linear.detach().cpu().numpy().tolist(),
        'predictors': data_provider.data.predictors.to_json()
    }

    with open(f'../../results/baseline_reconstructed_bdclim_safran_2023-2024.nc.json', 'w') as file:
        json.dump(results, file, indent=4)

if __name__ == "__main__":
    main()

