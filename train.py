from dataset import HW3Dataset
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv
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
train_mask = dataset[0].train_mask
val_mask = dataset[0].val_mask


#normalize train node year
subset = node_year[train_mask]
min_val = torch.min(subset)
max_val = torch.max(subset)
normalized_subset = 2*(subset - min_val) / (max_val - min_val) - 1
normalized_subset = normalized_subset.to(node_year.dtype)
node_year[train_mask] = normalized_subset

#normalize test node year
subset = node_year[val_mask]
min_val = torch.min(subset)
max_val = torch.max(subset)
normalized_subset = 2*(subset - min_val) / (max_val - min_val) - 1
normalized_subset = normalized_subset.to(node_year.dtype)
node_year[val_mask] = normalized_subset

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

model = GAT(hidden_channels=12, heads=15)
model = model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()



def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.to(device), edge_index.to(device))
    loss = criterion(out[train_mask].to(device),
                     y[train_mask].squeeze().to(device))
    loss.backward()
    optimizer.step()
    pred = out.argmax(dim=1).detach().cpu()
    correct = pred[train_mask] == y[train_mask].squeeze()
    acc = int(correct.sum()) / int(len(train_mask))

    return loss.detach().cpu(), acc


def test(mask):
    model.eval()
    out = model(data.to(device), edge_index.to(device))
    pred = out.argmax(dim=1).detach().cpu()
    correct = pred[mask] == y[mask].squeeze()
    loss = criterion(out[val_mask].detach().cpu(),
                     y[val_mask].squeeze())
    acc = int(correct.sum()) / int(len(mask))
    return acc, loss

def save_pred(mask):
    model.eval()
    out = model(data.to(device), edge_index.to(device))
    pred = out.argmax(dim=1).detach().cpu()
    return pred[mask]


acc_train = []
acc_val = []
loss_train = []
loss_val = []

for epoch in range(1, 500):
    loss, acc = train()
    val_acc, val_loss = test(val_mask)
    acc_train.append(acc)
    acc_val.append(val_acc)
    loss_train.append(loss)
    loss_val.append(val_loss)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train_Acc: {acc:.4f}, Val_Acc: {val_acc:.4f}')
pred_val = save_pred(val_mask)
pred_train = save_pred(train_mask)

acc_train = np.array(acc_train)
np.save('acc_train.npy', acc_train)
acc_val = np.array(acc_val)
np.save('acc_val.npy', acc_val)
loss_train = np.array(loss_train)
np.save('loss_train.npy', loss_train)
loss_val = np.array(loss_val)
np.save('loss_val.npy', loss_val)
y_val = np.array(y[val_mask])
np.save('y_val.npy', y_val)
y_train = np.array(y[train_mask])
np.save('y_train.npy', y_train)
pred_val = np.array(pred_val)
np.save('pred_val.npy', pred_val)
pred_train = np.array(pred_train)
np.save('pred_train.npy', pred_train)

torch.save(model, 'model.pth')
