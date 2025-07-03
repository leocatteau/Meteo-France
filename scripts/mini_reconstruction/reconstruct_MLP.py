import json
import pickle

import sys
sys.path.append('../..')

########################################################################
from models.MLP import MLP
from training.inference import Filler

from types import SimpleNamespace


def main():
    data_kwargs = SimpleNamespace()
    data_kwargs.root_path = '../../../datasets/'
    data_kwargs.data_path = 'train_test_period_masked.nc'
    data_kwargs.corrupted_data_path = 'train_test_period_masked.nc'
    data_kwargs.ideal = False

    model_kwargs = dict(hidden_dim=64)

    filler_kwargs = SimpleNamespace()
    filler_kwargs.mask_proba = 0.0
    filler_kwargs.mask_length = 0
    filler_kwargs.window = 24*1*1
    filler_kwargs.overlap = False
    filler_kwargs.model_path = '../../../results/mini_reconstruction/model/MLP_diffusion.pt'

    filler = Filler(data_kwargs, MLP, model_kwargs, filler_kwargs)
    original_data, corrupted_data, predictors, mask, eval_mask, reconstructed_data, RMSE, MAE, RG_RMSE, RG_MAE = filler.reconstruct()

    losses = {
        'RMSE': RMSE.tolist(),
        'MAE': MAE.tolist(),
        'RG_RMSE': RG_RMSE.tolist(),
        'RG_MAE': RG_MAE.tolist(),
    }

    print("Reconstruction completed. Saving results...")
    with open(f'../../../results/mini_reconstruction/data/reconstruction_losses_MLP.pkl', 'wb') as file:
        pickle.dump(losses, file)

    with open(f'../../../results/mini_reconstruction/data/reconstructed_data_MLP.pkl', 'wb') as file:
        pickle.dump(reconstructed_data.tolist(), file)

    with open(f'../../../results/mini_reconstruction/data/original_data_MLP.pkl', 'wb') as file:
        pickle.dump(original_data.tolist(), file)

    with open(f'../../../results/mini_reconstruction/data/corrupted_data_MLP.pkl', 'wb') as file:
        pickle.dump(corrupted_data.tolist(), file)

    with open(f'../../../results/mini_reconstruction/data/mask_MLP.pkl', 'wb') as file:
        pickle.dump(mask.tolist(), file)

    with open(f'../../../results/mini_reconstruction/data/predictors_MLP.json', 'w') as file:
        json.dump(predictors.to_json(), file)

    print("Reconstruction saved.")

if __name__ == "__main__":
    main()

