# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 14:13:48 2021

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
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import glob
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import rotate
import pickle
from fcn32 import fcn32
from aslloss import ASLSingleLabel
import cv2
from tqdm import tqdm
from resnet import resnet_fcn
#%%

class getdata(Dataset):
    def __init__(self , root , transform = None):
        self.root = root
        self.transform = transform
        self.filenames = []
        self.file_list = []
        with open(self.root+"\\roi.pkl", 'rb') as f:
            file = pickle.load(f)
        for ff in file.items():
            self.file_list.append(ff)
        self.length = len(file)
       
        self.transform_ = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomAffine(degrees=15,scale=(0.8,1)),
            # transforms.RandomRotation(15,resample=Image.NEAREST),
            transforms.ToTensor()
            ])
    
    def creat_mask(self , img , idx):
        mask = np.zeros_like(img)
        for x,y,w,h in idx:
            mask[ y:y+h ,x:x+w ] = 1
        return mask
    
    
    def __getitem__(self,index):
        img = np.load(self.root+"\\"+self.file_list[index][0]+".npy" , allow_pickle= True)
        mask = self.creat_mask(img , self.file_list[index][1][0])
        

        
        if self.transform :  
            img = cv2.resize(img , (128 , 768) , interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(mask , (128 , 768) , interpolation=cv2.INTER_NEAREST)
            r1 ,r2 = np.random.rand(2)
            if r1 > 0.5:
                img = np.flipud(img)
                mask = np.flipud(mask)
            if r2 < 0.5:
                img = np.fliplr(img)
                mask = np.fliplr(mask)
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)
        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)

        return img , mask
    
    def __len__(self):
        return self.length
    
trainset_path = r"D:\data\dl_SSD"
batchsize = 1
trainset = getdata(root = trainset_path , transform = True)  #true : complex , #false :float
trainset_loader = DataLoader(trainset , batch_size = batchsize, shuffle = True)

#%%
# dataiter = iter(trainset_loader)
# image, ri_distribute = dataiter.next()

#%%

# model = fcn32().to("cuda")
model = resnet_fcn().to("cuda")
# y = fcn(torch.randn(1, 1, 768,128))

# critirion = ASLSingleLabel().to("cuda")
critirion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

if os.path.exists(r"D:\lab\CODE\deep_learn\fcn32\resnet_fcn32.pt"):
    checkpoint = torch.load(r"D:\lab\CODE\deep_learn\fcn32\resnet_fcn32.pt")
    model.load_state_dict(checkpoint['state_dict'])


def meaniou(pred , label):
    intersect = pred + label
    intersect_area = torch.sum(intersect != 0).item()
    cross_area = torch.sum(intersect == 2).item()
    
    if torch.sum(intersect == 2) == torch.sum(label == 1):
        iou = 1
    
    elif cross_area == 0 and intersect_area == 0:
        iou = 1
    else :
        iou = cross_area / intersect_area     
    return iou
    
def pixel_acc(pred, target):
    correct = torch.sum(pred == target).item()
    total   = torch.sum(target == target).item()
    return correct / total

def train():
    epoch = 20
   
    for e in range(epoch):
        tot_loss = 0
        tot_iou = 0
        count = 0
        best_iou = 0
        for data , target in tqdm(trainset_loader):    
                     
            count+=1
            data , target = data.to("cuda") , target.to("cuda").long()
            
            model.zero_grad()
            output = model(data)
            loss = critirion(output , target[:,0,:,:])
            loss.backward()
            optimizer.step()
            p_out = output.data.max(1)[1].data
            
            tot_loss += loss.item()
            tot_iou += meaniou(p_out[0,:,:] , target[0,0,:,:])
            
            if count %10 == 0:
                plt.imshow(p_out[0,:,:].cpu().numpy())
                plt.title(str(count))
                plt.show()
                plt.imshow(target[0,0,:,:].cpu().numpy())
                plt.title(str(count))
                plt.show() 
            
        print("+++++++++++++++++++++++++++++++++++++++")    
        print("epoch : " + str(e))
        print("loss : " + str(tot_loss/count))
        tot_iou = tot_iou/count
        print("iou : " + str(tot_iou))
        print("+++++++++++++++++++++++++++++++++++++++")   
    
        if tot_iou > best_iou :
            best_iou = tot_iou
            save_path = r"D:\lab\CODE\deep_learn\fcn32\resnet_fcn32.pt"
            state = {"state_dict" : model.state_dict()}
            torch.save(state,save_path)

def test(model, image=None):
    model = model.eval()
    
    with torch.no_grad():
        if image is not None :
            image = image.to("cuda")
            output = model(image)
            p_out = output.data.max(1)[1].data.cpu().numpy()
            plt.imshow(p_out[0,:,:])
            plt.show()
        
        else :
            tot_iou = 0
            tot_picacc = 0
            count = 0
            for data , target in tqdm(trainset_loader):    
                count+=1
                data , target = data.to("cuda") , target.to("cuda").long()
                
                model.zero_grad()
                output = model(data)
        
                p_out = output.data.max(1)[1].data
                
                tot_iou += meaniou(p_out[0,:,:], target[0,0,:,:])
                tot_picacc += pixel_acc(p_out[0,:,:], target[0,0,:,:])
                
        
            print(tot_iou/count)
            print(tot_picacc / count)
  
  
train()
# test(model)
 
#%% 
# import time
# t1 = time.time()
# def to_model(img) :
#     size = max(img.shape)
#     img = cv2.blur(img , (3,3))
#     pad_num = 3072 - size
#     img = np.pad(img,((pad_num//2 , pad_num//2),(0,0)))
#     img = cv2.resize(img , (128 , 768) , interpolation=cv2.INTER_CUBIC).astype(np.float32)
#     img = torch.Tensor(img).unsqueeze(0).unsqueeze(1)
#     return img

# img = cv2.imread(r"D:\data\2021-02-03\12\test\c.bmp" , 0)
# # img = img[: , 200 : 712]

# img = to_model(img)

# test(model , img)
# t2 = time.time()
# print(t2-t1)
