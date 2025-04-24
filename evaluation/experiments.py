import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from tqdm import tqdm

from data_provider.data_provider import DataProvider
from models.baseline import mean_fill, svd, linear

def masking_robustness_baseline(args):
    probas = np.linspace(0.01, 0.95, 20)

    MSE_mean = np.zeros(len(probas))
    MSE_svd = np.zeros(len(probas))
    MSE_linear = np.zeros(len(probas))
    for i, p in enumerate(tqdm(probas)):
        args.mask_proba = p

        data_provider = DataProvider(args)
        X_train, y_train = data_provider.train_dataset[:]
        X_test, y_test = data_provider.test_dataset[:]

        mean_model = mean_fill(columnwise=True)
        y_mean = mean_model(torch.FloatTensor(X_test))
        MSE_mean[i] = mean_squared_error(y_mean, y_test)

        svd_model = svd()
        svd_model.train(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        y_svd = svd_model.predict(torch.FloatTensor(X_test))
        MSE_svd[i] = mean_squared_error(y_svd, y_test)

        linear_model = linear()
        linear_model.train(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        y_linear = linear_model.predict(torch.FloatTensor(X_test))
        MSE_linear[i] = mean_absolute_error(y_linear, y_test)

    return probas, MSE_mean, MSE_svd, MSE_linear