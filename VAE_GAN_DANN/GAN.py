# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:20:00 2020
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
@author: YX
"""
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset , DataLoader
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import glob
from PIL import Image
from tqdm import tqdm


class hwdata(Dataset):
    def __init__(self , root , transform = None):
        self.root = root
        self.transform = transform
        self.images = None
        self.filenames = []

        for filename in sorted(glob.glob(self.root + r"/*.png" ), key=os.path.getmtime):
            self.filenames.append(filename)
            
        self.length = len(self.filenames)
        
    
    def _transform(self, img):   
        rand_flip = np.random.rand()
        if rand_flip < 0.3:
            img = np.fliplr(img)
        elif rand_flip > 0.7:
            img = np.flipud(img)
        img = transforms.ToTensor()(img.copy()).float()
        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        img = transforms.ColorJitter(brightness=0.5, contrast=0.5)(img)
        return img
    
    def __getitem__(self,index):

        image_path = self.filenames[index]
        
        image = Image.open(image_path)
        
        if self.transform :     
            image = self._transform(image)
        else:
            image = transforms.ToTensor()(image.copy()).float()
        

        return image
    
    def __len__(self):
        return self.length



batchsize = 400
nz = 64
# trainset_path = r"C:\Users\YH\Desktop\CVDL\HW3\hw3-yohschang\hw3_data\face\train"
# trainset = hwdata(root = trainset_path , transform = True) 
# trainset_loader = DataLoader(trainset , batch_size = batchsize, shuffle = True)

#%%
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d( 64,512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 512 x 4 x 4
            nn.ConvTranspose2d(512,256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256 x 8 x 8
            nn.ConvTranspose2d(256,128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 x 16 x 16
            nn.ConvTranspose2d(128,64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64 x 32 x 32
            nn.ConvTranspose2d(64,3, 4, 2, 1, bias=False),
            nn.Tanh()
            # 3 x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

g_model = Generator().to("cuda")
g_model.apply(weights_init)
# print(g_model)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 3 x 64 x 64
            nn.Conv2d(3,64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 32 x 32
            nn.Conv2d(64,128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 16 x 16
            nn.Conv2d(128,256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 8 x 8
            nn.Conv2d(256,512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


d_model = Discriminator().to("cuda")
d_model.apply(weights_init)
# print(d_model)
criterion = nn.BCELoss()

d_optimizer = optim.Adam(d_model.parameters(), lr=0.001, betas=(0.5, 0.999))
g_optimizer = optim.Adam(g_model.parameters(), lr=0.001, betas=(0.5, 0.999))

def generate_input(batchsize = batchsize,z_dim = nz):
    # sample = torch.normal(0.5,0.5,size = (batchsize,z_dim,1,1))#gaussian
    sample = torch.randn(batchsize, z_dim,1,1)#uniform
    return sample

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr

#%%

def train():
# For each epoch
    for epoch in range(100):
        d_totloss = 0
        g_totloss = 0
        # For each batch in the dataloader
        if epoch > 0 and epoch % 15 == 0:
            set_learning_rate(d_optimizer ,0.6)
            set_learning_rate(g_optimizer ,0.6)
        for  data in tqdm(trainset_loader):
    
            d_model.zero_grad()
    
            data = data.to("cuda")
            
            real_label = torch.full((batchsize,), 1, dtype=torch.float).to("cuda")
            # real_label = torch.FloatTensor(batchsize,).uniform_(0.8,1).to("cuda") 
            real_output = d_model(data).view(-1)
            d_real_loss = criterion(real_output, real_label)
            d_real_loss.backward()
            D_x = real_output.mean().item()
            noise =generate_input().to("cuda")
            # noise = torch.randn(batchsize, nz, 1, 1, device=device)
            fake_in = g_model(noise)
            fake_label = torch.full((batchsize,), 0, dtype=torch.float).to("cuda")
            # fake_label = torch.FloatTensor(batchsize ,).uniform_(0,0.2).to("cuda")
    
            fake_output = d_model(fake_in.detach()).view(-1)
    
            d_fake_loss = criterion(fake_output, fake_label)
    
            d_fake_loss.backward()
            D_G_z1 = fake_output.mean().item()
    
            d_totloss += (d_real_loss.item() + d_fake_loss.item())
    
            d_optimizer.step()
    
            #generator
            
            g_model.zero_grad()
            g_label= torch.full((batchsize,), 1, dtype=torch.float).to("cuda") # fake labels are real for generator cost
            # g_label = torch.FloatTensor(batchsize,).uniform_(0.8,1).to("cuda")
            
            g_output = d_model(fake_in).view(-1)
    
            g_loss = criterion(g_output, g_label)
    
            g_loss.backward()
            D_G_z2 = g_output.mean().item()
    
            g_optimizer.step()
            g_totloss += g_loss.item()
    
        print("+++++++++++++++++++++++++++++++++++++++")
        print("epoch = "  +str(epoch))
        print("g_loss = "  +str(g_totloss/len(trainset_loader.dataset)))
        print("d_loss = "  +str(d_totloss/len(trainset_loader.dataset)))
        print("+++++++++++++++++++++++++++++++++++++++")


#%%
# train()
#%%

import os
if os.path.exists(r"hw3_2_model.pth"):
    checkpoint = torch.load(r"hw3_2_model.pth")
    d_model.load_state_dict(checkpoint["d_state_dict"])
    g_model.load_state_dict(checkpoint["g_state_dict"])
    d_optimizer.load_state_dict(checkpoint["d_optimizer"])
    g_optimizer.load_state_dict(checkpoint["g_optimizer"])

import sys
img_save_path = sys.argv[1]

import torchvision
torch.manual_seed(1000)

noise = torch.randn(32, nz, 1, 1, device="cuda")
# noise = generate_input(batchsize = 32).to("cuda")
fake = g_model(noise)
import matplotlib.pyplot as plt
def show(img,path):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg , (1,2,0)))
    plt.axis("off")
    plt.savefig(path)
show(torchvision.utils.make_grid(fake.cpu()),img_save_path)

#%%
# save_path = r"hw3_2_model.pth"
# state = {"d_state_dict" : d_model.state_dict(),"g_state_dict" : g_model.state_dict(),
#           "d_optimizer":d_optimizer.state_dict(),"g_optimizer":g_optimizer.state_dict()}
# torch.save(state,save_path)


