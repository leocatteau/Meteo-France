import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

from data_provider.data_factory import data_provider
from models.baseline import mean_fill, svd, linear

def masking_robustness(args, model=None, physics = False):
    probas = np.linspace(0.01, 0.9, 40)
    MSE_mean = {'mean': np.zeros(len(probas)), 'max': np.zeros(len(probas))}
    MSE_svd = {'mean': np.zeros(len(probas)), 'max': np.zeros(len(probas))}
    MSE_linear = {'mean': np.zeros(len(probas)), 'max': np.zeros(len(probas))}
    for i, p in enumerate(tqdm(probas)):
        args.mask_proba = p
        dataset, data_loader = data_provider(args, 'test')
        X, y = dataset[:]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

        mean_model = mean_fill(columnwise=True)
        y_mean = mean_model.predict(X_test)
        MSE_mean['mean'][i] = mean_squared_error(y_mean, y_test)
        if physics:
            MSE_mean['mean'][i] = mean_absolute_error(y_mean, y_test)
            MSE_mean['max'][i] = np.mean(np.max(np.abs(y_mean-y_test), axis=0))

        svd_model = svd()
        svd_model.train(X_train, y_train)
        y_svd = svd_model.predict(X_test)
        MSE_svd['mean'][i] = mean_squared_error(y_svd, y_test)
        if physics:
            MSE_svd['mean'][i] = mean_absolute_error(y_svd, y_test)
            MSE_svd['max'][i] = np.mean(np.max(np.abs(y_svd-y_test), axis=0))

        linear_model = linear()
        linear_model.train(X_train, y_train)
        y_linear = linear_model.predict(X_test)
        MSE_linear['mean'][i] = mean_absolute_error(y_linear, y_test)
        if physics:
            MSE_linear['mean'][i] = mean_absolute_error(y_linear, y_test)
            MSE_linear['max'][i] = np.mean(np.max(np.abs(y_linear-y_test), axis=0))


    return probas, MSE_mean, MSE_svd, MSE_linear