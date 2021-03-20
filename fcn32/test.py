# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 14:13:48 2021

@author: YX
"""
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset , DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
from tqdm import tqdm
import pickle
from fcn32 import fcn32
from aslloss import ASLSingleLabel
import cv2
from tqdm import tqdm
from resnet import resnet_fcn
import time
#%%

class testdata(Dataset):
    def __init__(self , root):
        self.root = root
        self.file_list = sorted(glob.glob(self.root + "\*.bmp") , key=os.path.getmtime)
        self.length = len(self.file_list)
    
    
    def __getitem__(self,index):
        img = cv2.imread(self.file_list[index], 0)[:,:512]
        img = cv2.blur(img , (3,3))
        img = cv2.resize(img , (128,768) , cv2.INTER_CUBIC)
        img = img.astype(np.float32)
        img = transforms.ToTensor()(img)
        return img 
    
    def __len__(self):
        return self.length
    
testset_path = r"D:\data\2021-03-02\4(2ms)"
batchsize = 20
testset = testdata(root = testset_path)  #true : complex , #false :float
testset_loader = DataLoader(testset , batch_size = batchsize, shuffle = False)

#%%
# dataiter = iter(trainset_loader)
# image, ri_distribute = dataiter.next()

#%%

# model = fcn32().to("cuda")
model = resnet_fcn().to("cuda")

if os.path.exists(r"D:\lab\CODE\deep_learn\fcn32\resnet_fcn32.pt"):
    print("load pretrain")
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

def test(model, image=None):
    model = model.eval()
    mask_stack = torch.tensor([])
    with torch.no_grad():
        if image is not None :
            image = image.to("cuda")
            output = model(image )
            p_out = output.data.max(1)[1].data.cpu().numpy()
            plt.imshow(p_out[0,:,:])
            plt.show()
        
        else :
            tot_iou = 0
            tot_picacc = 0
            count = 0
            for data in tqdm(testset_loader):  

                count+=1
                data= data.to("cuda") 

                output = model(data)
        
                p_out = output.data.max(1)[1].data.cpu()
                # plt.imshow(p_out[0,:,:])
                # plt.show()
                mask_stack = torch.cat((mask_stack , p_out) , dim = 0)
              
        return mask_stack

mask_stack = test(model)
savepath = testset_path+r"\analysis"

try:
    os.mkdir(savepath)
    np.save(savepath+r"\mask_stack.npy" , mask_stack)
    print("===mask save===")
except :
    np.save(savepath+r"\mask_stack.npy" , mask_stack)
    print("===create fail===")
    

 
#%% 


# def to_model(img) :
#     size = max(img.shape)
#     img = cv2.blur(img , (3,3))
#     pad_num = 3072 - size
#     img = np.pad(img,((pad_num//2 , pad_num//2),(0,0)))
#     img = cv2.resize(img , (128 , 768) , interpolation=cv2.INTER_CUBIC).astype(np.float32)
#     img = torch.Tensor(img).unsqueeze(0).unsqueeze(1)
#     return img

# def get_img(path):
#     for c , i in enumerate(sorted(glob.glob(path + "\*.bmp") , key=os.path.getmtime)):
#         if c < 10:
#             img = cv2.imread(i, 0)[:,:512]
#             img = cv2.blur(img , (3,3))
#             img = cv2.resize(img , (128,768) , cv2.INTER_CUBIC)
#             yield img
#         else :
#             break

# # img = cv2.imread(r"D:\data\2021-02-03\12\test\c.bmp" , 0)
# # # img = img[: , 200 : 712]
# # img = to_model(img)

# for img in get_img(path = r"D:\data\2021-02-03\12"):
#     img = torch.Tensor(img).unsqueeze(0).unsqueeze(1).to("cuda")
#     t1 = time.time()
#     test(model , img)
#     t2 = time.time()
#     print(t2-t1)
