print("Script Starting...")

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

data_path = r"/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Interns_2024/projects/single_cell_xenium_pred/final_workspace/poc/data/"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device", device)

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
      
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(256 * 4 * 4, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 256 * 4 * 4)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x

model = CNN().to(device)
# model.load_state_dict(torch.load('initial_cnn_model_weights.pth'))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1, verbose = True)

# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, threshold=5e-06)
      
num_epochs = 500
early_stop_count = 0
min_val_loss = float('inf')
patience = 15
all_train_losses = []
all_val_losses = []

print("Training Starting...")
      
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for data in tqdm(train_loader, total = len(train_loader)):
#         data = data.to(device)
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        # Print statistics
        running_train_loss += loss.item()
    
    avg_train_loss = running_train_loss/len(train_loader)
    all_train_losses.append(avg_train_loss)
    
    model.eval()
    running_val_loss = 0.0
    for data in tqdm(val_loader, total = len(val_loader)):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        running_val_loss += loss.item()
        
    avg_val_loss = running_val_loss/len(val_loader)
    all_val_losses.append(avg_val_loss)
    
#     scheduler.step(avg_val_loss)
    
    if avg_val_loss < min_val_loss:
        min_val_loss = avg_val_loss
        early_stop_count = 0
        
        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_val_loss}
        
        torch.save(checkpoint, '/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Interns_2024/projects/single_cell_xenium_pred/final_workspace/models/scratch_cnn_lr_sched.pth') #save checkpoint
        print("Model Checkpoint Saved")
    else:
        early_stop_count += 1
        
    if early_stop_count > patience:
        print("Early Stopping Activated")
        break
    
    with open('cnn_train_losses.txt', 'w') as f:
        for loss in all_train_losses:
            f.write(f"{loss}\n")
            
    with open('cnn_val_losses.txt', 'w') as f:
        for loss in all_val_losses:
            f.write(f"{loss}\n")
        
    print(f'Epoch: [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')
    
print("Training Complete")