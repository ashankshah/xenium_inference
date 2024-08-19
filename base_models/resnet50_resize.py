print("Starting Script...")

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
print("Device:", device)

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
    
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
    
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
        
        if self.transform:
            cell = self.transform(cell.astype(np.uint8))
            
        cell = cell/255.0
            
        cell = torch.tensor(cell, dtype=torch.float32).permute(2,0,1)
        vector = torch.tensor(vector, dtype=torch.float32)

        return cell, vector

train_dataset = custom_dataset(train_cells_arr, train_gene_exp, transform = transform)
val_dataset = custom_dataset(val_cells_arr, val_gene_exp, transform = transform)
test_dataset = custom_dataset(test_cells_arr, test_gene_exp, transform = transform)

train_loader = DataLoader(train_dataset, batch_size = 32)
val_loader = DataLoader(val_dataset, batch_size = 32)
test_loader = DataLoader(test_dataset, batch_size = 32)

for batch in train_loader:
    cell, gene_exp = batch
    print(cell.shape)
    print(gene_exp.shape)
    break

model_ft = models.resnet50(pretrained=True)
# model_ft.load_state_dict("finetuned_resnet18_model_weights.pth")
#need to make false? not competely sure but need to load the saved state dict
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_ftrs, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 10),
               )
model_ft = model_ft.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model_ft.parameters(), lr=0.00001)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, threshold = 5e-06)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1, verbose = True)

num_epochs = 500
patience = 15
early_counter = 0
min_val_loss = float('inf')

all_val_losses = []
all_train_losses = []

for epoch in range(num_epochs):
    model_ft.train()
    running_train_loss = 0.0
    for images, labels in tqdm(train_loader, total = len(train_loader)):
        images, labels = images, labels
        images = images.to(device).permute(0, 2, 1, 3)
        labels = labels.to(device)
   
        optimizer.zero_grad()
        
        outputs = model_ft(images)
        loss = criterion(outputs, labels)
    
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item() 
    
    avg_train_loss = running_train_loss/len(train_loader)
    all_train_losses.append(avg_train_loss)
    
    model_ft.eval()
    running_val_loss = 0.0
    for images, labels in tqdm(val_loader, total = len(val_loader)):
        images, labels = images, labels
        images = images.to(device).permute(0, 2, 1, 3)
        labels = labels.to(device)
        
        outputs = model_ft(images)
        loss = criterion(outputs, labels)
        
        running_val_loss += loss.item()
    
    avg_val_loss = running_val_loss/len(val_loader)
    all_val_losses.append(avg_val_loss)
    
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')
    
    if avg_val_loss < min_val_loss:
        min_val_loss = avg_val_loss
        early_counter = 0
        torch.save({
                    'epoch': epoch,
                    'model_state_dict':model_ft.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'loss':avg_val_loss
                    }, r"/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Interns_2024/projects/single_cell_xenium_pred/final_workspace/models/resnet50_resize.pth")
        print("Model Checkpoint Saved")
    else:
        early_counter += 1
    if early_counter > patience:
        print("Early Stopping Activated")
        break
    
    with open('resnet50_resize_train_losses.txt', 'w') as f:
        for loss in all_train_losses:
            f.write(f"{loss}\n")
            
    with open('resnet50_resize_val_losses.txt', 'w') as f:
        for loss in all_val_losses:
            f.write(f"{loss}\n")

print("Finished Training")