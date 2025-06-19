import json

import sys
sys.path.append('../..')

########################################################################
from models.MLP import MLP
from trainer.Filler import Filler

from types import SimpleNamespace


def main(mask_proba = 0.5):
    data_kwargs = SimpleNamespace()
    data_kwargs.root_path = '../../../datasets/'
    data_kwargs.data_path = 'bdclim_safran_2022-2024.nc'
    data_kwargs.ideal = True

    model_kwargs = dict(hidden_dim=64)

    filler_kwargs = SimpleNamespace()
    filler_kwargs.corrupt = True
    filler_kwargs.mask_proba = mask_proba
    filler_kwargs.mask_length = 24*7*3
    filler_kwargs.window = 24*1*1
    filler_kwargs.model_path = '../../../results/noise_level_MLP/model/MLP_diffusion.pt'

    filler = Filler(data_kwargs, MLP, model_kwargs, filler_kwargs)
    corrupted_data, predictors, mask, eval_mask, reconstructed_data, loss = filler.reconstruct()

    results = {
        # 'original_data': corrupted_data.tolist(),
        # 'predictors': predictors.to_json(),
        # 'mask': mask.tolist(),
        # 'eval_mask': eval_mask.tolist(),
        # 'reconstructed_data': reconstructed_data.tolist(),
        'loss': loss.tolist(),
    }

    print("Reconstruction completed. Saving results...")
    with open(f'../../../results/noise_level_MLP/data/reconstruction_diffusion_mask_proba_{mask_proba}.json', 'w') as file:
        json.dump(results, file, indent=4)
    print("Reconstruction saved.")

if __name__ == "__main__":
    mask_proba = float(sys.argv[1])
    main(mask_proba=mask_proba)

