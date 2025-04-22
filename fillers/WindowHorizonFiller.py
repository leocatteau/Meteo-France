from tqdm import tqdm
import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.optim.lr_scheduler import CosineAnnealingLR
from models.baseline import mean_fill

class WindowHorizonFiller:
    def __init__(self, model, model_kwargs, args):
        self.model = model(**model_kwargs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=0.0001)
        self.loss = nn.MSELoss()
        self.epochs = args.epochs
        self.keep_proba = args.keep_proba

    def predict(self, batch):
        # include the preprocess 
        x = batch.pop('x').float()
        mean_model = mean_fill(columnwise=True)
        x_mean = mean_model(x)
        mask = batch.pop('mask')

        prediction = self.model(x_mean, mask, **batch)
        return prediction

    def training_step(self, batch):
        # get the target
        y = batch.pop('y').float()

        # compute prediction and loss
        y_hat, _ = self.predict(batch)
        loss = self.loss(y_hat, y)

        self.optimizer.zero_grad()
        loss.backward()  
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()
    
    def test_step(self, batch):
        # get the target
        y = batch.pop('y').float()

        # compute prediction and loss
        y_hat = self.predict(batch)
        loss = self.loss(y_hat, y)
        return loss.item()
    
    def train(self, train_dataloader, test_dataloader, verbose=False):
        train_losses = []
        test_losses = []
        for epoch in tqdm(range(self.epochs)):
            train_loss = 0.0
            self.model.train()
            for batch in train_dataloader:
                loss = self.training_step(batch)
                train_loss += loss
            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)

            test_loss = 0.0
            self.model.eval()
            self.model.train(False)
            with torch.no_grad():
                for batch in test_dataloader:
                    loss = self.test_step(batch)
                    test_loss += loss
            test_loss /= len(test_dataloader)
            test_losses.append(test_loss)

            if verbose:
                print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}")
        return train_losses, test_losses
    
    def impute_dataset(self, dataloader, data_provider):
        self.model.eval()
        self.model.train(False)

        x, y, mask = [], [], []
        for batch in dataloader:
            x_, y_, mask_ = batch.pop('x').view(-1, data_provider.data.n_nodes, 1), batch.pop('y').view(-1, data_provider.data.n_nodes, 1), batch.pop('mask').view(-1, data_provider.data.n_nodes, 1)
            x.append(x_)
            y.append(y_)
            mask.append(mask_)
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)
        mask = torch.cat(mask, dim=0)

        with torch.no_grad():
            mean_model = mean_fill(columnwise=True)
            x_mean = mean_model(x)
            x = x_mean[None,:,:,:].float()
            mask = mask[None,:,:,:].byte()

            prediction = self.model(x, mask)
            prediction = prediction.squeeze().cpu().numpy()
            print(f"Imputed data shape: {prediction.shape}")
        return prediction

        # y = []
        # with torch.no_grad():
        #     for batch in dataloader:
        #         y_hat = self.predict(batch).squeeze().cpu().numpy()
        #         y_hat = y_hat.reshape(-1, y_hat.shape[-1])
        #         print(f"Imputed batch shape: {y_hat.shape}")
        #         y.append(y_hat)
        # y = np.concatenate(y, axis=0)
        # print(f"Imputed data shape: {y.shape}")
        # return y

    def trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()