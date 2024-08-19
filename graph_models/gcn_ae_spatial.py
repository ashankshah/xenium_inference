print("Starting Autoencoder GCN Script")

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
import h5py
from torch_geometric.data import Data, DataLoader, InMemoryDataset
from torch_geometric.utils import dense_to_sparse
from sklearn.metrics.pairwise import euclidean_distances
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.nn.norm import LayerNorm
from torch.nn import Linear
from itertools import combinations
from PIL import Image
from torch_geometric.nn import knn_graph
from torch_geometric.utils import add_remaining_self_loops
import networkx as nx
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torchvision
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

data_path = r"/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Interns_2024/projects/single_cell_xenium_pred/final_workspace/poc/data/"
df = pd.read_parquet(data_path + "spatially_var_10_patched_df.parquet")

df_train = df[df['set'] == 'train']
df_val = df[df['set'] == 'val']
df_test = df[df['set'] == 'test']

with h5py.File(data_path + "meansize_cells.h5", 'r') as hf:
    cells_arr = np.array(hf['cells'])

print("Data Loaded")

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
    

feature_extractor = cell_autoencoder()
model_path = '/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Interns_2024/projects/single_cell_xenium_pred/final_workspace/models/'
checkpoint = torch.load(model_path + "autoencoder.pth", map_location = device)
feature_extractor.load_state_dict(checkpoint['model_state_dict'])
feature_extractor = feature_extractor.to(device)

latent_length = 128
k = 10
                            
train_graphs = []
for patch_index in tqdm(list(set(df_train['patch_index'])), total = len(list(set(df_train['patch_index'])))):
    temp_df = df_train[df_train['patch_index'] == patch_index]
    temp_df = temp_df.reset_index()
                        
    temp_cell_embeds = np.zeros((len(temp_df), latent_length))
    temp_cell_coords = np.zeros((len(temp_df), 2))

    for i, row in temp_df.iterrows():
        cell_tensor = ((torch.from_numpy(cells_arr[row['arr_index']].transpose(2,0,1)).unsqueeze(0).float())/255.0).to(device)
#         print(cell_tensor.shape)
        cell_latent = feature_extractor.encoder(cell_tensor)[0]
        temp_cell_embeds[i] = cell_latent.cpu().detach().numpy()
        temp_cell_coords[i] = np.array((row['x_centroid'], row['y_centroid']))
    
    temp_cell_coords = torch.tensor(temp_cell_coords, dtype=torch.float)
    node_features = torch.tensor(temp_cell_embeds)
    exp = torch.tensor(list(temp_df['gene_exp_vector']))

    G = knn_graph(temp_cell_coords, k=k, batch=None, loop=True)
    G = G.detach().cpu()
    G = add_remaining_self_loops(G)[0]
                        
    temp_graph_data = Data(x = node_features, edge_index = G.cpu(), gene_exp = exp)
    train_graphs.append(temp_graph_data)

val_graphs = []
for patch_index in tqdm(list(set(df_val['patch_index'])), total = len(list(set(df_val['patch_index'])))):
    temp_df = df_val[df_val['patch_index'] == patch_index]
    temp_df = temp_df.reset_index()
                        
    temp_cell_embeds = np.zeros((len(temp_df), latent_length))
    temp_cell_coords = np.zeros((len(temp_df), 2))

    for i, row in temp_df.iterrows():
        cell_tensor = ((torch.from_numpy(cells_arr[row['arr_index']].transpose(2,0,1)).unsqueeze(0).float())/255.0).to(device)
        cell_latent = feature_extractor.encoder(cell_tensor)[0]
        temp_cell_embeds[i] = cell_latent.cpu().detach().numpy()
        temp_cell_coords[i] = np.array((row['x_centroid'], row['y_centroid']))
    
    temp_cell_coords = torch.tensor(temp_cell_coords, dtype=torch.float)
    node_features = torch.tensor(temp_cell_embeds)
    exp = torch.tensor(list(temp_df['gene_exp_vector']))

    G = knn_graph(temp_cell_coords, k=k, batch=None, loop=True)
    G = G.detach().cpu()
    G = add_remaining_self_loops(G)[0]
                        
    temp_graph_data = Data(x = node_features, edge_index = G.cpu(), gene_exp = exp)
    val_graphs.append(temp_graph_data)
                            
test_graphs = []
for patch_index in tqdm(list(set(df_test['patch_index'])), total = len(list(set(df_test['patch_index'])))):
    temp_df = df_test[df_test['patch_index'] == patch_index]
    temp_df = temp_df.reset_index()
                        
    temp_cell_embeds = np.zeros((len(temp_df), latent_length))
    temp_cell_coords = np.zeros((len(temp_df), 2))

    for i, row in temp_df.iterrows():
        cell_tensor = ((torch.from_numpy(cells_arr[row['arr_index']].transpose(2,0,1)).unsqueeze(0).float())/255.0).to(device)
        cell_latent = feature_extractor.encoder(cell_tensor)[0]
        temp_cell_embeds[i] = cell_latent.cpu().detach().numpy()
        temp_cell_coords[i] = np.array((row['x_centroid'], row['y_centroid']))
    
    temp_cell_coords = torch.tensor(temp_cell_coords, dtype=torch.float)
    node_features = torch.tensor(temp_cell_embeds)
    exp = torch.tensor(list(temp_df['gene_exp_vector']))

    G = knn_graph(temp_cell_coords, k=k, batch=None, loop=True)
    G = G.detach().cpu()
    G = add_remaining_self_loops(G)[0]
                        
    temp_graph_data = Data(x = node_features, edge_index = G.cpu(), gene_exp = exp)
    test_graphs.append(temp_graph_data)
                            
print(len(train_graphs))
print(len(val_graphs))
print(len(test_graphs))

def prepare_data(data):
    data.x = data.x.float()
    data.edge_index = data.edge_index.long()
    data.y = data.gene_exp.float()
    return data
                            
class graph_dataset(InMemoryDataset):
    def __init__(self, graphs):
        self.graphs = graphs
        super(graph_dataset, self).__init__()

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        data = self.graphs[idx]
        return prepare_data(data)
                            
train_dataset = graph_dataset(train_graphs)
val_dataset = graph_dataset(val_graphs)
test_dataset = graph_dataset(test_graphs)
                            
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=6, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=True)
                            
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(LayerNorm(hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(LayerNorm(hidden_channels))

        self.mlp = torch.nn.Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.num_layers):
            x_res = x
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x + x_res) 

        x = self.mlp(x)
        return x
                            
# Hyperparameters
in_channels = 128  #latent rep
hidden_channels = 128
out_channels = 10  #gene exp vector len
num_layers = 3
learning_rate = 0.001
num_epochs = 300
                            
model = GNN(in_channels, hidden_channels, out_channels, num_layers).to(device)
criterion = torch.nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
                            
min_val_loss = float('inf')
early_stopping_counter = 0
patience = 10
                            
all_train_losses = []
all_val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for data in tqdm(train_loader, total = len(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        
        loss = criterion(out, data.y) 
        running_train_loss += loss
        
        loss.backward()
        optimizer.step()
    
    avg_train_loss = running_train_loss/len(train_loader)
    all_train_losses.append(avg_train_loss)
    
    model.eval()
    
    running_val_loss = 0.0
    for data in tqdm(val_loader, total = len(val_loader)):
        data = data.to(device)
        out = model(data)
        
        loss = criterion(out, data.y)
        running_val_loss += loss
    
    avg_val_loss = running_val_loss/len(val_loader)
    all_val_losses.append(avg_val_loss)
    
    scheduler.step(avg_val_loss)
    
    if avg_val_loss < min_val_loss:
        min_val_loss = avg_val_loss
        early_stopping_counter = 0 
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, '/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Interns_2024/projects/single_cell_xenium_pred/final_workspace/models/ae_gcn_spatial.pth')
        print("Saved Model Checkpoint")
    
    else:
        early_stopping_counter += 1

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')
    
    if early_stopping_counter > patience:
        print("Early Stopping Activated")
        break
                            
    with open('ae_gcn_train_losses.txt', 'w') as f:
        for loss in all_train_losses:
            f.write(f"{loss}\n")
            
    with open('ae_gcn_val_losses.txt', 'w') as f:
        for loss in all_val_losses:
            f.write(f"{loss}\n")


print("Training Completed")