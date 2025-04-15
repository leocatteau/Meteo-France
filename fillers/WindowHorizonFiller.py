from tqdm import tqdm
import torch 
import torch.nn as nn
import numpy as np

from torch.optim.lr_scheduler import CosineAnnealingLR

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
        mask = batch.pop('mask')
        prediction = self.model(x, mask, **batch)
        return prediction

    def training_step(self, batch):
        # get the data
        y = batch.pop('y').float()

        # Extract mask and target
        mask = batch['mask'].clone().detach()
        batch['mask'] = torch.bernoulli(mask.clone().detach().float() * self.keep_proba).byte()

        # compute prediction and loss
        y_hat, _ = self.predict(batch)
        loss = self.loss(y_hat, y)

        self.optimizer.zero_grad()
        loss.backward()  
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()
    
    def test_step(self, batch):
        # get the data
        y = batch.pop('y').float()

        # Extract mask and target
        mask = batch['mask'].clone().detach()
        batch['mask'] = torch.bernoulli(mask.clone().detach().float() * self.keep_proba).byte()

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
            with torch.no_grad():
                for batch in test_dataloader:
                    loss = self.test_step(batch)
                    test_loss += loss
            test_loss /= len(test_dataloader)
            test_losses.append(test_loss)

            if verbose:
                print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}")
        return train_losses, test_losses
    
    def impute_dataset(self, dataloader):
        self.model.eval()
        y = []
        with torch.no_grad():
            for batch in dataloader:
                y_hat = self.predict(batch).view(-1, batch['y'].shape[2], 1)
                y.append(y_hat)
        y = torch.cat(y, dim=0).squeeze(2).cpu().numpy()
        return y

    def trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)