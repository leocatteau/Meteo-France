from copy import deepcopy
from tqdm import tqdm
import torch 
import torch.nn as nn

class WindowHorizonFiller:
    def __init__(self, model, model_kwargs, args):
        self.model = model(**model_kwargs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args.lr)
        self.loss = nn.MSELoss()
        self.masking_proba = args.masking_proba

    def predict(self, batch):
        # include the preprocess 
        x = batch.pop('x').float()
        prediction = self.model(x, **batch)
        # imputation = prediction * batch['eval_mask'] + x * (1 - batch['eval_mask'])
        return prediction

    def training_step(self, batch):
        # get the data
        y = batch.pop('y').float()

        # extract mask
        mask = batch['mask'].clone().detach()
        batch['mask'] = torch.bernoulli(mask.clone().detach().float() * self.masking_proba).byte()
        eval_mask = batch['mask'] - mask

        # compute prediction and loss
        y_hat, _ = self.predict(batch)
        loss = self.loss(y_hat, y)

        self.optimizer.zero_grad()
        loss.backward()  
        self.optimizer.step()
        # self.scheduler.step()
        return loss.item()
    
    def test_step(self, batch):
        # get the data
        y = batch.pop('y').float()

        # extract mask
        mask = batch['mask'].clone().detach()
        batch['mask'] = torch.bernoulli(mask.clone().detach().float() * self.masking_proba).byte()
        eval_mask = batch['mask'] - mask

        # compute prediction and loss
        y_hat = self.predict(batch)
        loss = self.loss(y_hat, y)
        return loss.item()
    
    def train(self, train_dataloader, test_dataloader, epochs, verbose=False):
        train_losses = []
        test_losses = []
        for epoch in tqdm(range(epochs)):
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
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}")
        return train_losses, test_losses
        

    def trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)