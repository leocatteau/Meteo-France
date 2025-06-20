import json

import sys
sys.path.append('../..')

########################################################################
from models.MLP import MLP
from training.inference import Filler

from types import SimpleNamespace


def main():
    data_kwargs = SimpleNamespace()
    data_kwargs.root_path = '../../../datasets/'
    data_kwargs.data_path = 'test_period.nc'
    data_kwargs.corrupted_data_path = 'test_period_masked.nc'
    data_kwargs.ideal = False

    model_kwargs = dict(hidden_dim=64)

    filler_kwargs = SimpleNamespace()
    filler_kwargs.mask_proba = 0.0
    filler_kwargs.mask_length = 0
    filler_kwargs.window = 24*1*1
    filler_kwargs.model_path = '../../../results/mini_reconstruction/model/MLP_diffusion.pt'

    filler = Filler(data_kwargs, MLP, model_kwargs, filler_kwargs)
    original_data, corrupted_data, predictors, mask, eval_mask, reconstructed_data, loss, RG_loss = filler.reconstruct()

    results = {
        'original_data': original_data.tolist(),
        'corrupted_data': corrupted_data.tolist(),
        'predictors': predictors.to_json(),
        'mask': mask.tolist(),
        'reconstructed_data': reconstructed_data.tolist(),
        'loss': loss.tolist(),
        'RG_loss': RG_loss.tolist(),
    }

    print("Reconstruction completed. Saving results...")
    with open(f'../../../results/mini_reconstruction/data/reconstruction_diffusion_MLP.json', 'w') as file:
        json.dump(results, file, indent=4)
    print("Reconstruction saved.")

if __name__ == "__main__":
    main()

