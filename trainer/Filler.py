# import time
import numpy as np
import torch 
import torch.nn as nn
import time

from torch.optim.lr_scheduler import CosineAnnealingLR
from models.baseline import mean_fill

import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image
import cv2

from trainer.custom_losses import masked_MSE, spatiotemporal_masked_MSE, temporal_gradient_MSE, spatial_graph_gradient_MSE, spatial_laplacian_MSE, mixed_loss


class Filler():
    def __init__(self, model, model_kwargs, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: ", self.device)
        self.model = model(**model_kwargs).to(self.device)
        self.mean_model = mean_fill(columnwise=True).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=0.0001)
        # self.loss = nn.MSELoss()
        self.train_loss = spatiotemporal_masked_MSE
        self.loss = masked_MSE
        self.epochs = args.epochs
        self.keep_proba = args.keep_proba
        self.spatial_weight = args.spatial_weight
        self.eta = args.eta
        # self.graph = args.graph

    def predict(self, batch):
        # include the preprocess 
        x = batch.pop('x').float()
        mask = batch.pop('mask')

        x_mean = self.mean_model(x,mask)
        # [b s n c]
        prediction = self.model(x_mean, mask, **batch)
        return prediction

    def training_step(self, batch):
        # get the target
        y = batch.pop('y').float()
        eval_mask = batch.pop('eval_mask')

        # compute prediction and loss
        y_hat, prediction = self.predict(batch)
        
        # loss = self.loss(y_hat, y) # evaluate on all data (only with clean data)
        loss = self.train_loss(y_hat, y, eval_mask, spatial_weight=self.spatial_weight) # evaluate on the artificial mask only
        # loss = self.loss(prediction, y, mask^eval_mask) # train the model to predict all the signal (avoiding the original mask), fonctopnne très mal en test sur la tâche réellement attendue

        self.optimizer.zero_grad()
        loss.backward()  
        self.optimizer.step()

        loss = self.loss(y_hat, y, eval_mask) # evaluate on the artificial mask only
        return loss.item()
    
    def test_step(self, batch):
        # get the target
        y = batch.pop('y').float()
        eval_mask = batch.pop('eval_mask')

        # compute prediction and loss
        y_hat = self.predict(batch)
        # loss = self.loss(y_hat, y) # evaluate on all data (only with clean data)
        loss = self.loss(y_hat, y, eval_mask) # evaluate on the artificial mask only for testing
        return loss.item()
    
    def train(self, train_dataloader, test_dataloader):
        early_stopping = EarlyStopping(tolerance=10, overfit_delta=1, saturation_delta=1e-3)
        start_time = time.time()
        print(f"start training")

        train_losses = []
        test_losses = []
        for epoch in range(self.epochs):
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
            # early_stopping(train_losses, test_losses)
            # if early_stopping.early_stop:
                # break
            self.scheduler.step()
            print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}, time: {time.time() - start_time:.2f}s")
            #print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}")
        return train_losses, test_losses
    
    def reconstruct(self,data, mask):
        self.model.eval()
        self.model.train(False)

        # [s n] -> [b s n c]
        x = data.unsqueeze(0).unsqueeze(-1).to(self.device)
        mask = mask.unsqueeze(0).unsqueeze(-1).to(self.device)
        batch = {'x': x, 'mask': mask}

        with torch.no_grad():
            y_hat = self.predict(batch)
            y_hat = y_hat.squeeze().cpu()#.numpy()
        return y_hat
    
    def reconstruct_from_loader(self,dataloader, get_original_data=False):
        self.model.eval()
        self.model.train(False)

        original = []
        reconstructed = []
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                assert batch['x'].shape[0] == 1, "batch size should be 1"
                original.append(batch['x'].squeeze().cpu())
                # if i > 0:
                #     batch = {torch.cat([old_batch[k], batch[k]], dim=0) for k in batch.keys()}.to(self.device)
                #     old_batch = {k: batch[k][overlap:] for k in batch.keys()}.to(self.device)

                y_hat = self.predict(batch)
                # y_hat = y_hat[overlap:].squeeze().cpu()#.numpy()
                y_hat = y_hat.squeeze().cpu()#.numpy()
            reconstructed.append(y_hat)
        original = torch.cat(original, dim=0)
        reconstructed = torch.cat(reconstructed, dim=0)

        if get_original_data:
            return original, reconstructed
        return reconstructed

    def latent_training(self, train_dataloader, data, mask):
        start_time = time.time()
        print(f"start mini training for latent space analysis")

        losses = []
        video_frames = []
        for epoch in range(self.epochs):
            loss = 0.0
            self.model.train()
            for batch in train_dataloader:
                loss += self.training_step(batch)
            loss /= len(train_dataloader)
            losses.append(loss)
            self.scheduler.step()
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.8f}, time: {time.time() - start_time:.2f}s")

            data = data.clone().unsqueeze(0).unsqueeze(-1).to(self.device)
            mask = mask.clone().unsqueeze(0).unsqueeze(-1).to(self.device)
            eval_batch = {'x': data, 'mask': mask}
            with torch.no_grad():
                imputation, prediction = self.predict(eval_batch)
                imputation = imputation.squeeze().cpu()
                prediction = prediction.squeeze().cpu()
                data = data.squeeze()
                mask = mask.squeeze()

                # load into video 
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(imputation.T, xticklabels=False, yticklabels=False, cmap="coolwarm", ax=ax)
                ax.set_title("Latent prediction")
                ax.set_xlabel("Time")
                ax.set_ylabel("Station")
                buf = BytesIO()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                img = Image.open(buf)
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                video_frames.append(frame)

        # save video
        height, width, _ = video_frames[0].shape
        video = cv2.VideoWriter('../latent_training.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
        for frame in video_frames:
            video.write(frame)
        video.release()
        return losses


    def trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
        self.model.eval()

    # def impute_dataset(self, data_provider):
    #     data = data_provider.dataset.pytorch()
    #     mask = torch.FloatTensor(data_provider.dataset.mask)

    #     # [s n] -> [b s n c]
    #     x = data[None,:,:,None]
    #     mask = mask[None,:,:,None].byte()
        
    #     batch = {'x': x, 'mask': mask}

    #     with torch.no_grad():
    #         y_hat = self.predict(batch)
    #         y_hat = y_hat.squeeze().cpu().numpy()
        
    #     return y_hat

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