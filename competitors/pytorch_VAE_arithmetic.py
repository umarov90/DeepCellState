#!/usr/bin/env python
# coding: utf-8

# In[1]:


# prerequisites
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data):
        'Initialization'
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.data[index]





bs = 200
x_dim = 978

mcf7_tr = pd.read_csv("./tr_ts_data/MCF7_tr.csv", header=None).values
pc3_tr = pd.read_csv("./tr_ts_data/PC3_tr.csv", header=None).values
test_data = pd.read_csv("./tr_ts_data/PC3_ts.csv", header=None).values

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=Dataset(np.vstack((mcf7_tr,pc3_tr))), 
    batch_size=bs, shuffle=False)
mcf7_tr_loader = torch.utils.data.DataLoader(dataset=Dataset(mcf7_tr), 
    batch_size=bs, shuffle=False)
pc3_tr_loader = torch.utils.data.DataLoader(dataset=Dataset(pc3_tr), 
    batch_size=bs, shuffle=False)


test_loader = torch.utils.data.DataLoader(dataset=Dataset(test_data), 
    batch_size=bs, shuffle=False)


# In[2]:


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        
        self.x_dim = x_dim

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, self.x_dim))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

# build model
vae = VAE(x_dim=x_dim, h_dim1= 512, h_dim2=256, z_dim=20)
if torch.cuda.is_available():
    vae.cuda()


optimizer = optim.Adam(vae.parameters())

# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, x_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.cuda().float()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))



def test():
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data in test_loader:
            data = data.cuda().float()
            recon, mu, log_var = vae(data)
            
            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, 500):
    train(epoch)
    test()

# calculate delta
mcf7_tr_data = torch.tensor(mcf7_tr).cuda().float()
_, mu_mcf7_tr, _ = vae(mcf7_tr_data)
pc3_tr_data = torch.tensor(pc3_tr).cuda().float()
_, mu_pc3_tr, _ = vae(pc3_tr_data)


# compute the predicted result for the testing data
test_data = torch.tensor(test_data).cuda().float()
_, mu_test, _ = vae(test_data)
predicted = vae.decoder(mu_test+torch.mean(mu_mcf7_tr-mu_pc3_tr, 0))

np.savetxt("MCF7_ts_pred.csv", predicted.cpu().detach().numpy(), delimiter=",")

