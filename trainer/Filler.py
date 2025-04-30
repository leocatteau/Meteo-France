# import time
import numpy as np
import torch 
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingLR
from models.baseline import mean_fill


class Filler():
    def __init__(self, model, model_kwargs, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: ", self.device)
        self.model = model(**model_kwargs).to(self.device)
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

        # [b s n c]
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
        return loss.item()
    
    def test_step(self, batch):
        # get the target
        y = batch.pop('y').float()

        # compute prediction and loss
        y_hat = self.predict(batch)
        loss = self.loss(y_hat, y)
        return loss.item()
    
    def train(self, train_dataloader, test_dataloader):
        early_stopping = EarlyStopping(tolerance=5, overfit_delta=1, saturation_delta=1e-3)

        train_losses = []
        test_losses = []
        for epoch in range(self.epochs):
            # start_time = time.time()
            # print(f"start training, time: {start_time:.2f}s")
            print(f"start training")
            train_loss = 0.0
            self.model.train()
            for batch in train_dataloader:
                #batch = {k: v.to(device) for k, v in batch.items()}
                loss = self.training_step(batch)
                train_loss += loss
            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)

            test_loss = 0.0
            self.model.eval()
            self.model.train(False)
            with torch.no_grad():
                for batch in test_dataloader:
                    #batch = {k: v.to(device) for k, v in batch.items()}
                    loss = self.test_step(batch)
                    test_loss += loss
            test_loss /= len(test_dataloader)
            test_losses.append(test_loss)

            # early stopping
            early_stopping(train_losses, test_losses)
            if early_stopping.early_stop:
                break
            self.scheduler.step()
            # print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}, time: {time.time() - start_time:.2f}s")
            print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}")
        return train_losses, test_losses

    def impute_dataset(self, data_provider):
        data = data_provider.dataset.pytorch()
        mask = torch.FloatTensor(data_provider.dataset.mask)

        # [s n] -> [b s n c]
        x = data[None,:,:,None]
        mask = mask[None,:,:,None].byte()
        
        batch = {'x': x, 'mask': mask}

        with torch.no_grad():
            y_hat = self.predict(batch)
            y_hat = y_hat.squeeze().cpu().numpy()
        
        return y_hat


    def trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    # def impute_dataloader(self, dataloader, data_provider):
    #     self.model.eval()
    #     self.model.train(False)

    #     x, y, mask = [], [], []
    #     for batch in dataloader:
    #         x_, y_, mask_ = batch.pop('x').view(-1, data_provider.data.n_nodes, 1), batch.pop('y').view(-1, data_provider.data.n_nodes, 1), batch.pop('mask').view(-1, data_provider.data.n_nodes, 1)
    #         x.append(x_)
    #         y.append(y_)
    #         mask.append(mask_)
    #     x = torch.cat(x, dim=0)
    #     y = torch.cat(y, dim=0)
    #     mask = torch.cat(mask, dim=0)

    #     with torch.no_grad():
    #         mean_model = mean_fill(columnwise=True)
    #         x_mean = mean_model(x)
    #         x = x_mean[None,:,:,:].float()
    #         mask = mask[None,:,:,:].byte()

    #         prediction = self.model(x, mask)
    #         prediction = prediction.squeeze().cpu().numpy()
    #         print(f"Imputed data shape: {prediction.shape}")
    #     return prediction



class EarlyStopping:
    def __init__(self, tolerance=5, overfit_delta=0, saturation_delta=0):

        self.tolerance = tolerance
        self.overfit_delta = overfit_delta
        self.saturation_delta = saturation_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss[-1] - train_loss[-1]) > self.overfit_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
                print("overfitting")

        if len(train_loss) > self.tolerance:
            if np.abs(train_loss[-1] - train_loss[-self.tolerance]) < self.saturation_delta:
                self.early_stop = True
                print("saturation")