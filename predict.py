from torch.autograd._functions import tensor

from dataset import HW3Dataset
import torch
from dataset import HW3Dataset
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
torch.cuda.empty_cache()

dataset = HW3Dataset(root='data/hw3/')
data1 = dataset[0].x
edge_index = dataset[0].edge_index
y = dataset[0].y
node_year = dataset[0].node_year

#normalize train node year
subset = node_year
min_val = torch.min(subset)
max_val = torch.max(subset)
normalized_subset = 2*(subset - min_val) / (max_val - min_val) - 1
normalized_subset = normalized_subset.to(node_year.dtype)
node_year = normalized_subset

data = torch.cat((data1, node_year), dim=1)

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads):
        super().__init__()
        torch.manual_seed(1234567)
        self.norm = torch.nn.BatchNorm1d(dataset.num_features + 1)
        self.conv1 = GATConv(dataset.num_features + 1, out_channels=hidden_channels, heads=heads)
        self.conv2 = GATConv(in_channels=hidden_channels*heads, out_channels=dataset.num_classes, heads=heads)

    def forward(self, x, edge_index):
        x = self.norm(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = torch.load('model.pth')

def test():
    model.eval()
    out = model(data.to(device), edge_index.to(device))
    pred = out.argmax(dim=1).detach().cpu()
    return pred

pred = test()
print(pred)

import csv
import torch

csv_file = 'prediction.csv'

tensor_list = pred.tolist()

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(['idx', 'prediction'])

    for i, value in enumerate(tensor_list):
        writer.writerow([i, value])
