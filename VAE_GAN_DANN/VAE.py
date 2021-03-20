# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 12:58:17 2020

@author: YX
https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/variational_autoencoder.ipynb#scrollTo=mZaVrj0hX1ry
https://github.com/atinghosh/VAE-pytorch/blob/master/VAE_celeba.py
"""

import torchvision
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset , DataLoader
from PIL import Image
import glob
import numpy as np
# import time
import os
import pandas as pd

#%% read dataset

class hwdata(Dataset):
    def __init__(self , root , transform = None , label = False):
        self.root = root
        self.label = label
        self.transform = transform
        self.images = None
        self.filenames = []

        df = pd.read_csv(r'C:\Users\YH\Desktop\CVDL\HW3\hw3-yohschang\hw3_data\face\test.csv')

        for filename in sorted(glob.glob(self.root + r"/*.png" ), key=os.path.getmtime):
            
            if self.label:
                img_name = filename.replace(root,"")[1:]
                idx = df[df.image_name == img_name].index.tolist()
                target = int(df.loc[idx[0],["Male"]][0])
                # print(target)
                self.filenames.append((filename,target))
            else : 
                self.filenames.append(filename)
            
        self.length = len(self.filenames)
        
    
    def _transform(self, img):   
        rand_flip = np.random.rand()
        if rand_flip < 0.3:
            img = np.fliplr(img)
        elif rand_flip > 0.7:
            img = np.flipud(img)
        img = transforms.ToTensor()(img.copy()).float()
        # img = transforms.ColorJitter(brightness=0.5, contrast=0.5)(img)
        return img
    
    def __getitem__(self,index):
        if self.label:
            image_path , target = self.filenames[index]
        else:
            image_path = self.filenames[index]
        
        image = Image.open(image_path)
        
        if self.transform :     
            image = self._transform(image)
        else:
            image = transforms.ToTensor()(image.copy()).float()
        
        if self.label:
            # print(target)
            return image , target
        else:
            return image
    
    def __len__(self):
        return self.length

#%%
# trainset_path = r"C:\Users\YH\Desktop\CVDL\HW3\hw3-yohschang\hw3_data\face\train"
# trainset = hwdata(root = trainset_path , transform = True , label = False) 
# trainset_loader = DataLoader(trainset , batch_size = 400, shuffle = True)

# testset_path = r"C:\Users\YH\Desktop\CVDL\HW3\hw3-yohschang\hw3_data\face\test"
# testset = hwdata(root = testset_path , transform = False , label = False) 
# testset_loader = DataLoader(testset , batch_size = 10, shuffle = True)

# dataiter = iter(trainset_loader)
# images  = dataiter.next() 

# import matplotlib.pyplot as plt
# def show(img):
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg , (1,2,0)))
# show(torchvision.utils.make_grid(images))

#%% VAE model
#H dim : full fig dim ; z dim : latent space dim = sqrt(h dim)
class Encoder(nn.Module):
    def __init__(self , label = False):
        self.label = label
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1) # out: 64*32*32
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 128, kernel_size=4, stride=2, padding=1) # out: 128*16*16
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128,256, kernel_size=4, stride=2, padding=1) # out: 256*8*8
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256,64, kernel_size=4, stride=2, padding=1) # out: 64*4*4
        self.relu4 = nn.ReLU()
        self.fc_mu = nn.Linear(in_features=64*4*4, out_features=256)
        self.fc_logvar = nn.Linear(in_features=64*4*4, out_features=256)
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        if self.label:
            return x ,  x_mu, x_logvar  #for tsne
        else:
            return  x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( 256,512, 4, 1, 0),
            nn.ReLU(True),
            # state size. 256 x 4 x 4
            nn.ConvTranspose2d(512,128, 4, 2, 1),
            nn.ReLU(True),
            # state size. 128 x 8 x 8
            nn.ConvTranspose2d(128,32, 4, 2, 1),
            nn.ReLU(True),
            # state size. 32 x 16 x 16
            nn.ConvTranspose2d(32,16, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16,3, 4, 2, 1),
            nn.Sigmoid(),
            # state size. 3x 64 x 64
        )
    def forward(self, input):
        return self.main(input)
    
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder(label = True)
        self.decoder = Decoder()
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent.view(latent.size()+(1,1)))
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    
def vae_loss(recon_x, x, mu, logvar):
    # recon_loss =  critirion(recon_x.view(-1, 4096), x.view(-1, 4096))
    # recon_loss = F.binary_cross_entropy(recon_x.view(-1, 4096), x.view(-1, 4096), reduction='sum')
    recon_loss = F.mse_loss(recon_x.view(-1, 4096), x.view(-1, 4096), reduction='mean')
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kldivergence /= (400 * 3 * 64 * 64)  #if set ruduction = mean
    return recon_loss + kldivergence , kldivergence


model = VAE().to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr
#%%
train_loss_list = []
kld_loss_list = []

# from tqdm import tqdm
def train(model , epoch):
    model = model.train()

    for e in range(epoch):
        if e > 0 and e % 20 == 0:
            set_learning_rate(optimizer ,0.6)
        train_loss = 0
        kld_loss = 0
        for images in tqdm(trainset_loader):
            # print(targets)
            images = images.to("cuda")
            recon_images, mu, logvar = model(images)
            loss , KLD =vae_loss(recon_images , images, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            kld_loss += KLD.item()
            
        print("==================================================")
        print("epoch = "  +str(e))
        print("loss(bce+kl) = "  +str(train_loss/len(trainset_loader.dataset)))
        print("KL_loss = "  +str(kld_loss/len(trainset_loader.dataset)))
        print("==================================================")
        train_loss_list.append(train_loss/len(trainset_loader.dataset))
        kld_loss_list.append(kld_loss/len(trainset_loader.dataset))

    

#%%

train(model , 50)
save_path = r"hw3_1_model.pth"
state = {"state_dict" : model.state_dict(),"optimizer":optimizer.state_dict()}
torch.save(state,save_path)

#%%
import matplotlib.pyplot as plt
plt.plot(kld_loss_list)
plt.show()
plt.plot(train_loss_list)
plt.show()

#%% question 3 
dataiter = iter(testset_loader)
images  = dataiter.next() 
model = model.eval()
images = images.to("cuda")
recon_images, mu, logvar = model(images)
recon_images = recon_images.cpu()
images = images.cpu()

mses = []
for recon , img in zip(recon_images , images):
    mse = np.sqrt(np.mean((recon.detach().numpy() - img.detach().numpy())**2))
    mses.append(mse)

import matplotlib.pyplot as plt
def show(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg , (1,2,0)))
    plt.axis("off")
    plt.show()
show(torchvision.utils.make_grid(images))
show(torchvision.utils.make_grid(recon_images))

#%%TSNE

import matplotlib.pyplot as plt
from matplotlib import cm
def plot_with_labels(lowDWeights, labels):
    # print(labels)
    plt.cla() #clear当前活动的坐标轴
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1] #把Tensor的第1列和第2列,也就是TSNE之后的前两个特征提取出来,作为X,Y
    for x, y, s in zip(X, Y, labels):
        # print(s)
        c = cm.rainbow(int(255 * s / 2))
        #plt.text(x, y, s, backgroundcolor=c, fontsize=9)
        plt.text(x, y, str(s),color=c,fontdict={'weight': 'bold', 'size': 9}) #在指定位置放置文本
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')

testset = hwdata(root = testset_path , transform = False , label = True) 
testset_loader = DataLoader(testset , batch_size = 400, shuffle = True)

count = 0
for images  , target in tqdm(testset_loader):
    # print(targets)
    images = images.to("cuda")
    encode_out,mu,_ = model.encoder(images)
    
    latent_embedded = TSNE(n_components=2).fit_transform(mu.cpu().detach().numpy())
    N = 10
    # print(latent_embedded[:, 0].shape , target.shape)
    for i , j , l in zip(latent_embedded[:, 0], latent_embedded[:, 1] , target.cpu().numpy()):
        # plt.figure(figsize=(8, 6))
        if l == 1:
            c= "r"
        else:
            c = "b"
        plt.scatter(i,j, c=c, marker='o', edgecolor='none')
        plt.title("Male")
    plt.show()

#%% load save model
import os
if os.path.exists(r"hw3_1_model.pth"):
    checkpoint = torch.load(r"hw3_1_model.pth")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

#%% create random face

import sys
img_save_path = sys.argv[1]
print(img_save_path)
torch.manual_seed(1234)

import matplotlib.pyplot as plt
def show(img,path):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg , (1,2,0)))
    plt.axis("off")
    plt.savefig(path)

z = torch.randn(32,256,1,1).to("cuda")
out = model.decoder(z)

show(torchvision.utils.make_grid(out.cpu()),img_save_path)
