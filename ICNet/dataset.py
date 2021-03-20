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
from glob import glob
from os import listdir
from os.path import isfile, join
from PIL import Image , ImageFilter
from tqdm import tqdm
from scipy.ndimage import rotate
import pickle
import cv2
import random
import cupy
#%%
img_size = 3072

def generate_bg(path):
    print("\n generating bg for " + path)
    bg = cupy.zeros((img_size , img_size))
    f = 0
    for c , i in tqdm(enumerate(sorted(glob(path + "/*.bmp"),key = os.path.getmtime))):
        if c % 10 == 0:
            f+=1
            img = cv2.blur(cv2.imread(i , 0) , (3,3))
            bg += cupy.array(img)
    bg /= f
    bg = cupy.asnumpy(bg)
    cv2.imwrite(path+"/bg.bmp" , bg)
    

class getdata(Dataset):
    def __init__(self , labelpath , datapath , transform = None):
        # self.labelpath = labelpath
        # self.datapath = datapath
        self.transform = transform
        self.filenames = []
        self.label_list = []
        self.data_list = []
        self.resize = 1024
        self.clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(20,20))
        self.folder_num = []
        self.bg = {}
        
        
        labelfiles = [join(labelpath, f) for f in listdir(labelpath)] 
        
        for c , l in enumerate(labelfiles):
            for d in sorted(glob(l+'/*.txt'),key = os.path.getmtime):
                self.label_list.append(d)
                number = l.split('_')[-1] + '_'
                imgnum = d.split('\\')[-1].replace("txt" , 'bmp')
                img_pt = l.replace(labelpath , datapath).replace('_','\\')+'\\'+number+imgnum
                self.data_list.append(img_pt)
                self.folder_num.append(c)
                
            bgpath = img_pt.replace(number+imgnum , "bg.bmp")
            if os.path.exists(bgpath):
                self.bg.update({c : bgpath})
            else:
                generate_bg(bgpath.replace("\\bg.bmp", ''))
                self.bg.update({c: bgpath})

       
        self.length = len(self.label_list)
        # print(self.length)
        
        self.transform_ = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=0,scale=(0.9,1),shear=(6, 9),resample = 3),
            transforms.RandomRotation(180,resample=Image.NEAREST),
            ])
    
    def creat_mask(self,  img , idx):
        shapes = img.shape   #[y , x]
    
        mask = np.zeros_like(img)
        if len(idx.shape) <= 1:
            try:
                _ ,x,y,w,h  = idx
                x,w = int(x*shapes[1]) , int((w*shapes[1])//2)
                y,h = int(y*shapes[0]) , int((h*shapes[0])//2)
                mask[ y-h:y+h ,x-w:x+w ] = 1
                return mask
            except :
                return mask
        
        else:
            for _ ,x,y,w,h in idx:
                x,w = int(x*shapes[1]) , int((w*shapes[1])//2)
                y,h = int(y*shapes[0]) , int((h*shapes[0])//2)
                mask[ y-h:y+h ,x-w:x+w ] = 1
            return mask
        
    
    def __getitem__(self,index):
        
        # number = self.root.split('\\')[-1] + '_'
        # label_p = self.label_list[index]
        # img_p = label_p.split('\\')[-1].replace("txt" , 'bmp')
        label_p = self.label_list[index]
        img_p = self.data_list[index]
        # print(img._p)
        # img = Image.open(img_p).filter(ImageFilter.BoxBlur(1))
        img = cv2.blur(cv2.imread(img_p,0),(3,3))
        mask = self.creat_mask(img , np.loadtxt(label_p))  
        
        bg_p = self.bg[self.folder_num[index]]
        # print(img_p)
        # print(bg_p)
        bg = cv2.imread(bg_p,0)
        img = img-bg*0.9
        img[img < 0] = 0
        
        if self.transform :  
            # img = transforms.Resize((self.resize,self.resize), Image.BICUBIC)(img)
            img = cv2.resize(img , (self.resize,self.resize), cv2.INTER_CUBIC)
            # img = transforms.ColorJitter(brightness=0.7, contrast=0.7)(img)
            # plt.imshow(img , cmap = "gray")
            # plt.show()
            
            mask = transforms.Resize((self.resize,self.resize), Image.NEAREST)(Image.fromarray(mask))

            stack = np.concatenate([img , mask]).reshape(-1,self.resize,self.resize)
           
            seed = np.random.randint(20210304)
            for idx , img in enumerate(stack):
                np.random.seed(seed)
                if np.random.rand() >= 0.5:
                    img = np.fliplr(img)
                    img = np.flipud(img)
                
                random.seed(seed)
                # img = self.transform_(img)
                stack[idx,:,:] = img
            
            # img = self.clahe.apply(stack[0,...]).astype(np.float32)
            img = stack[0,...].astype(np.float32)
            mask = stack[1,...].astype(np.float32)
        
        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)

        return img , mask
    
    def __len__(self):
        return self.length
    
    
class valdata(Dataset):
    def __init__(self , labelpath , datapath , transform = None):
        # self.labelpath = labelpath
        # self.datapath = datapath
        self.transform = transform
        self.filenames = []
        self.label_list = []
        self.data_list = []
        self.resize = 1024
    
        labelfiles = [join(labelpath, f) for f in listdir(labelpath)] 
        self.clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(20,20))
        for l in labelfiles:
            for d in sorted(glob(l+'/*.txt'),key = os.path.getmtime):
                self.label_list.append(d)
                number = l.split('_')[-1] + '_'
                imgnum = d.split('\\')[-1].replace("txt" , 'bmp')
                img_pt = l.replace(labelpath , datapath).replace('_','\\')+'\\'+number+imgnum
                self.data_list.append(img_pt)
       
        self.length = len(self.label_list)
        # print(self.length)
        
    def creat_mask(self,  img , idx):
        size = img.shape  #[y , x]
        
        mask = np.zeros_like(img)
        if len(idx.shape) <= 1:
            try:
                _ ,x,y,w,h  = idx
                x,w = int(x*size[1]) , int((w*size[1])//2)
                y,h = int(y*size[0]) , int((h*size[0])//2)
                mask[ y-h:y+h ,x-w:x+w ] = 1
                return mask
            except :
                return mask
        
        else:
            for _ ,x,y,w,h in idx:
                x,w = int(x*size[1]) , int((w*size[1])//2)
                y,h = int(y*size[0]) , int((h*size[0])//2)
                mask[ y-h:y+h ,x-w:x+w ] = 1
            return mask

    
    def __getitem__(self,index):
        
        label_p = self.label_list[index]
        img_p = self.data_list[index]
        img = cv2.imread(img_p,0)
        img = cv2.blur(img , (3,3))
        mask = self.creat_mask(img , np.loadtxt(label_p))   
             
        img = cv2.resize(img , (self.resize,self.resize) , interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask , (self.resize,self.resize) , interpolation=cv2.INTER_NEAREST)
              
        img = self.clahe.apply(img)
        img = transforms.ToTensor()(img.astype(np.float32))
        mask = transforms.ToTensor()(mask.astype(np.float32))

        return img , mask
    
    def __len__(self):
        return self.length
    
class testdata(Dataset):
    def __init__(self , root ):
        self.root = root
        self.file_list = []
        self.clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(20,20))
        for d in sorted(glob(self.root+'/*.bmp'),key = os.path.getmtime):
            self.file_list.append(d)
        
        self.length = len(self.file_list)
             
    
    def __getitem__(self,index):
        # img = Image.open(self.file_list[index]).filter(ImageFilter.BoxBlur(1))
        # img = transforms.Resize((768,768), Image.BICUBIC)(img)
 
        img = cv2.imread(self.file_list[index],0) 
        img = cv2.resize(img , (1024,1024) , interpolation=cv2.INTER_CUBIC)
        img = self.clahe.apply(img)
        img = img.astype(np.float32)
        img = transforms.ToTensor()(img)

        return img
    
    def __len__(self):
        return self.length
    

    
if __name__ == '__main__':
    
    labelpath = r"D:\data\dl_SSD\icn_label_data"
    datapath = r"D:\data"
    # trainset_path = r"D:\data\2021-03-03\1"
    batchsize = 1
    trainset = getdata(labelpath , datapath , transform = True)  #true : complex , #false :float
    
    trainset_loader = DataLoader(trainset , batch_size = batchsize, shuffle = True)
    
    # trainset_path = r"D:\data\dl_SSD"
    # batchsize = 1
    # trainset = getdata2(root = trainset_path , transform = True)  #true : complex , #false :float
    # trainset_loader = DataLoader(trainset , batch_size = batchsize, shuffle = True)
        
    
    dataiter = iter(trainset_loader)
    image, ri_distribute = dataiter.next()
    #%%
    plt.imshow(image[0,0,...],cmap = "gray")
    plt.show()  
    plt.imshow(ri_distribute[0,0,...])
    plt.show()  
