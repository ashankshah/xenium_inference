print("Starting Autoencoder Job")
import numpy as np
import torch
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import cv2
from tqdm import tqdm
import tifffile
import dask.dataframe as dd
import gzip
import shutil
import h5py
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import psutil
import torchvision
from torchvision import datasets, models, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = r"/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Interns_2024/projects/single_cell_xenium_pred/final_workspace/poc/data/"
df = pd.read_parquet(data_path + "spatially_var_10_patched_df.parquet")

train_df = df[df['set'] == 'train']
val_df = df[df['set'] == 'val']
test_df = df[df['set'] == 'test']

train_gene_exp = np.array(train_df['gene_exp_vector'])
val_gene_exp = np.array(val_df['gene_exp_vector'])
test_gene_exp = np.array(test_df['gene_exp_vector'])

with h5py.File(data_path + "meansize_cells.h5", 'r') as hf:
    cells_arr = np.array(hf['cells'])
    
train_cells_arr = np.zeros((len(train_df), 64, 64, 3))
val_cells_arr = np.zeros((len(val_df), 64, 64, 3))
test_cells_arr = np.zeros((len(test_df), 64, 64, 3))

index = 0
for i, row in tqdm(train_df.iterrows(), total = len(train_df)):
    train_cells_arr[index] = cells_arr[row['arr_index']]
    index += 1

index = 0
for i, row in tqdm(val_df.iterrows(), total = len(val_df)):
    val_cells_arr[index] = cells_arr[row['arr_index']]
    index += 1

index = 0
for i, row in tqdm(test_df.iterrows(), total = len(test_df)):
    test_cells_arr[index] = cells_arr[row['arr_index']]
    index += 1
    
class custom_dataset(Dataset):
    def __init__(self, cells, vectors, transform=None):
        self.cells = cells
        self.vectors = vectors
        self.transform = transform

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        cell = self.cells[idx]
        vector = self.vectors[idx]
        
        cell = cell/255.0
        
        if self.transform:
            image = self.transform(image)
            
        cell = torch.tensor(cell, dtype=torch.float32).permute(2,0,1)
        vector = torch.tensor(vector, dtype=torch.float32)

        return cell, vector

train_dataset = custom_dataset(train_cells_arr, train_gene_exp)
val_dataset = custom_dataset(val_cells_arr, val_gene_exp)
test_dataset = custom_dataset(test_cells_arr, test_gene_exp)

train_loader = DataLoader(train_dataset, batch_size = 32)
val_loader = DataLoader(val_dataset, batch_size = 32)
test_loader = DataLoader(test_dataset, batch_size = 32)

class cell_autoencoder(nn.Module):
    def __init__(self):
        super(cell_autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), 
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(), 
            nn.Linear(4*4*256, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 128)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 4*4*256),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
model = cell_autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1, verbose = True)

num_epochs = 500
patience = 10
best_loss = float('inf')
early_stop_counter = 0

all_train_losses = []
all_val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for data in tqdm(train_loader, total = len(train_loader), dynamic_ncols = True):
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    avg_loss = running_loss/len(train_loader)
    all_train_losses.append(avg_loss)
    
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for data in tqdm(val_loader, total = len(val_loader)):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            running_val_loss += loss.item()
    
    avg_val_loss = running_val_loss/len(val_loader)
    all_val_losses.append(avg_val_loss)
    
    scheduler.step(avg_val_loss)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss}, Validation Loss: {avg_val_loss}')
    
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        early_stop_counter = 0
        checkpoint = {
                'epoch': epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':best_loss
                }
        
        torch.save(checkpoint, r"/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Interns_2024/projects/single_cell_xenium_pred/final_workspace/models/autoencoder.pth")
        
        print("Model Checkpoint Saved")
    
    else:
        early_stop_counter +=1
        if early_stop_counter > patience:
            print("Early Stopping Activated")
            break
    
    with open('autoencoder_train_losses.txt', 'w') as f:
        for loss in all_train_losses:
            f.write(f"{loss}\n")
            
    with open('autoencoder_val_losses.txt', 'w') as f:
        for loss in all_val_losses:
            f.write(f"{loss}\n")

print("Training Complete")