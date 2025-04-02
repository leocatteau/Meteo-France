import torch 
import torch.nn as nn
from tqdm import tqdm

from utils.functions import torch_nan_to_num

def train(model, train_dataloader, test_dataloader, lr=0.001, epochs=100, verbose=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_losses = []
        test_losses = []
        for epoch in tqdm(range(epochs)):
            train_loss = 0.00
            for x, y in train_dataloader:
                x = x.float().to(device)
                y = y.float().to(device)

                mask = torch.isnan(x)
                y_pred = model(x)
                y_pred = y_pred*(mask) + torch_nan_to_num(x)
                loss = criterion(y_pred, y)
                optimizer.zero_grad()  
                loss.backward()        
                optimizer.step()       
                train_loss += loss.item()
            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)

            test_loss = 0.00
            with torch.no_grad():
                for x, y in test_dataloader:
                    x = x.float().to(device)
                    y = y.float().to(device)

                    mask = torch.isnan(x)
                    y_pred = model(x)
                    y_pred = y_pred*(mask) + torch_nan_to_num(x)
                    loss = criterion(y_pred, y)
                    test_loss += loss.item()
                test_loss /= len(test_dataloader)
                test_losses.append(test_loss)
            
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}")
        
        return train_losses, test_losses