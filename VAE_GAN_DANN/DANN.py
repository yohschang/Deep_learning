# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 22:29:31 2020
https://github.com/Yangyangii/DANN-pytorch/blob/master/DANN.ipynb
@author: YX
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
# from sklearn.manifold import TSNE
import pandas as pd
from torch.autograd import Function

#%%
class hwdata(Dataset):
    def __init__(self , root  , dfpath = 0, transform = None ):
        self.root = root
        self.transform = transform
        self.images = None
        self.filenames = []

        # df = pd.read_csv(dfpath)
        file_list = [file for file in os.listdir(root) if file.endswith('.png')]
        file_list.sort()

        for i, file in enumerate(file_list):
            filename = os.path.join(root, file)
        # for i in range(df["image_name"].size):
            # filename = self.root + "/" + str(df["image_name"][i])
            # label = int(df["label"][i])
            self.filenames.append((filename , file))
            # print(str(df["image_name"][i]) , label)
            # print(filename , label)
            
        self.length = len(self.filenames)
        
    
    def _transform(self, img):   
        # rand_flip = np.random.rand()
        # if rand_flip < 0.3:
        #     img = np.fliplr(img)
        # elif rand_flip > 0.7:
        #     img = np.flipud(img)
        img = transforms.ToTensor()(img.copy()).float()
        # img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)

        # img = transforms.ColorJitter(brightness=0.5, contrast=0.5)(img)
        return img
    
    def __getitem__(self,index):

        image_path , target = self.filenames[index]
        image = Image.open(image_path).convert('RGB')
        # image = image.convert('RGB')
        
        # if self.transform :     
        #     image = self._transform(image)
        if self.transform is not None :
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image.copy()).float()
        

        return image , target

    
    def __len__(self):
        return self.length
    
transform = transforms.Compose([
    # transforms.Grayscale(1),
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 100
    
# mnistm_pth = r"C:\Users\YH\Desktop\CVDL\HW3\hw3-yohschang\hw3_data\digits\mnistm"
# svhn_pth = r"C:\Users\YH\Desktop\CVDL\HW3\hw3-yohschang\hw3_data\digits\svhn"
# usps_pth = r"C:\Users\YH\Desktop\CVDL\HW3\hw3-yohschang\hw3_data\digits\usps"
# trainset_path = [mnistm_pth+"/train",svhn_pth+"/train",usps_pth+"/train"]
# train_label_path = [mnistm_pth+"/train.csv",svhn_pth+"/train.csv",usps_pth+"/train.csv"]
# testset_path = [mnistm_pth+"/test",svhn_pth+"/test",usps_pth+"/test"]
# test_label_path = [mnistm_pth+"/test.csv",svhn_pth+"/test.csv",usps_pth+"/test.csv"]

# source_trainset = hwdata(root = trainset_path[2] , dfpath = train_label_path[2] , transform = transform ) 
# source_trainset_loader = DataLoader(source_trainset , batch_size = batch_size, shuffle = False)

# target_trainset = hwdata(root = trainset_path[0] ,dfpath = train_label_path[0], transform = transform ) 
# target_trainset_loader = DataLoader(target_trainset , batch_size = batch_size, shuffle = False)


#%%

class FeatureExtractor(nn.Module):
    """
        Feature Extractor
    """
    def __init__(self, in_channel=3, hidden_dims=512):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, hidden_dims, 3, padding=1),
            nn.BatchNorm2d(hidden_dims),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        
    def forward(self, x):
        h = self.conv(x).squeeze() # (N, hidden_dims)
        return h

class Classifier(nn.Module):
    def __init__(self, input_size=512, num_classes=10):
        super(Classifier, self).__init__()
        # self.layer = nn.Sequential(
        #     nn.Linear(input_size, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, num_classes),
        # )
        self.linear1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(256, num_classes)
        
    def forward(self, h):
        h = self.relu1(self.linear1(h))
        c = self.linear2(h)
        return h ,c

class Discriminator(nn.Module):
    def __init__(self, input_size=512, num_classes=1):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, num_classes),
            nn.Sigmoid(),
        )
        # self.linear1 =  nn.Linear(256, 128)
        # self.lkr1 = nn.LeakyReLU(0.2)
        # self.linear2 = nn.Linear(128, num_classes)
        # self.sigmoid = nn.Sigmoid()
    def forward(self, h):
        y = self.layer(h)
        # tsne = self.lkr1(self.linear1(y))
        # y = self.sigmoid(self.linear2(tsne))
        return  y

F = FeatureExtractor().to("cuda")
C = Classifier().to("cuda")
D = Discriminator().to("cuda")
# print(D)
d_critirion = nn.BCELoss()
c_critirion = nn.CrossEntropyLoss()

F_opt = torch.optim.Adam(F.parameters(),lr = 0.0001)
C_opt = torch.optim.Adam(C.parameters(),lr = 0.0001)
D_opt = torch.optim.Adam(D.parameters(),lr = 0.0001)

#%%
max_epoch = 50


def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1+np.exp(-10.*p)) - 1.
def train(step = step):
    final_acc = 0
    ll_c, ll_d = [], []
    acc_list = []
    for epoch in range(1, max_epoch+1):
        for idx ,( source_data , target_data)  in enumerate(zip(source_trainset_loader , target_trainset_loader)):
            source_data , source_label =  source_data
            src, labels = source_data.to("cuda") , source_label.to("cuda")
            target_data , target_label =  target_data
            tgt = target_data.to("cuda")
            
            D_src = torch.ones(len(src), 1).to("cuda") # Discriminator Label to real
            D_tgt = torch.zeros(len(tgt), 1).to("cuda") # Discriminator Label to fake
            D_labels = torch.cat([D_src, D_tgt], dim=0)
                    
            input_data = torch.cat([src, tgt], dim=0)
            data_feature = F(input_data)
            data_domain = D(data_feature.detach())
            
            domain_loss = d_critirion(data_domain, D_labels)
            D.zero_grad()
            domain_loss.backward()
            D_opt.step()
            
            src_class = C(data_feature[:len(src)])
            data_domain = D(data_feature)
            class_loss = c_critirion(src_class, labels)
            domain_loss = d_critirion(data_domain, D_labels)
            lamda = 0.1*get_lambda(epoch, max_epoch)
            Ltot = class_loss -lamda*domain_loss
            
            F.zero_grad()
            C.zero_grad()
            D.zero_grad()
            
            Ltot.backward()
            
            C_opt.step()
            F_opt.step()
            
            if step % 100 == 0:
                # dt = datetime.datetime.now().strftime('%H:%M:%S')
                print('Epoch: {}/{}, Step: {}, D Loss: {:.4f}, C Loss: {:.4f}, lambda: {:.4f}'.format(epoch, max_epoch, step, domain_loss.item(), class_loss.item(), lamda))
                ll_c.append(class_loss)
                ll_d.append(domain_loss)
            step += 1
        acc = test(step)
        acc_list.append(acc)
        if acc> final_acc:
            final_acc = acc
            save_path = r"hw3_3_model(usps-mnist).pth"
            state = {"F_state_dict" : F.state_dict(),
                      "C_state_dict" : C.state_dict(),
                      "D_state_dict" : D.state_dict(),
                      "F_optimizer":F_opt.state_dict(),
                      "C_optimizer":C_opt.state_dict(),
                      "D_optimizer":D_opt.state_dict()}
            torch.save(state,save_path)
#%%
def test(step,pred_save_pth):
    F.eval()
    C.eval()
    image_id = []
    pred_label = []
    with torch.no_grad():
        # corrects = torch.zeros(1).to("cuda")
        # for idx, (src, labels) in enumerate(eval_loader):
        #     src, labels = src.to("cuda"), labels.to("cuda")
        #     c = C(F(src))
        #     _, preds = torch.max(c, 1)
        #     corrects += (preds == labels).sum()
        # acc = corrects.item() / len(eval_loader.dataset)
        # print('***** Eval Result: {:.4f}, Step: {}'.format(acc, step))
        
        corrects = torch.zeros(1).to("cuda")
        for tgt, labels in test_loader:
            tgt = tgt.to("cuda")
            # print(type(c.cpu()))
            h,c = C(F(tgt))

            _, preds = torch.max(c,1)
            # pred = c[0].max(1,keepdim = True)[1]
            # corrects += (preds == labels).sum()
            for img_id , im_pd in zip(labels , preds):
                # print(im_pd)
                image_id.append(img_id)
                pred_label.append(int(im_pd.cpu().numpy()))
        # acc = corrects.item() / len(test_loader.dataset)
        # print('***** Test Result: {:.4f}, Step: {}'.format(acc, step))
        dicts = {"image_name" :image_id , "label" : pred_label }
        DF = pd.DataFrame(dicts)
        DF.to_csv(pred_save_pth,index = 0)
    F.train()
    C.train()
    # return acc



#%%
# import os
# if os.path.exists(r"hw3_3_model(usps-mnist).pth"):
#     checkpoint = torch.load(r"hw3_3_model(usps-mnist).pth")
#     F.load_state_dict(checkpoint['F_state_dict'])
#     C.load_state_dict(checkpoint['C_state_dict'])
#     D.load_state_dict(checkpoint['D_state_dict'])
#     # F_opt.load_state_dict(checkpoint['F_optimizer'])
#     # C_opt.load_state_dict(checkpoint['C_optimizer'])
# train()

#%%
# testset_path = [mnistm,svhn,usps]
# evalset = hwdata(root = testset_path[1] ,dfpath = test_label_path[1] , transform = transform) 
# eval_loader = DataLoader(evalset , batch_size = batch_size, shuffle = False)

# testset = hwdata(root = testset_path[2] ,dfpath = test_label_path[2], transform = transform ) 
# test_loader = DataLoader(testset , batch_size = batch_size, shuffle = False)

#%%
import sys
test_dataset  = sys.argv[1]
domain = sys.argv[2]
output_path = sys.argv[3]

testset = hwdata(root = test_dataset, transform = None ) 
test_loader = DataLoader(testset , batch_size = batch_size, shuffle = False)

if domain == "mnistm":
    load_model = r"hw3_3_model_usps-mnist.pth"
elif domain == "usps":
    load_model = r"hw3_3_model_svhn-usps.pth"
elif domain == "svhn":
    load_model = r"hw3_3_model_mnist-svhn.pth"


import os
if os.path.exists(load_model):
    checkpoint = torch.load(load_model)
    F.load_state_dict(checkpoint['F_state_dict'])
    C.load_state_dict(checkpoint['C_state_dict'])
    D.load_state_dict(checkpoint['D_state_dict'])
    F_opt.load_state_dict(checkpoint['F_optimizer'])
    C_opt.load_state_dict(checkpoint['C_optimizer'])

test(1,output_path)

#%% TSNE
# def hook(module, inputdata, output):
#     return output.data
    
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from matplotlib import cm

# count = 0
# plt.cla()
# for (source_data , target_data)  in tqdm(zip(eval_loader , test_loader)):
#     source_data , source_label =  source_data
#     src, src_labels = source_data.to("cuda") , source_label.to("cuda")
#     target_data , target_label =  target_data
#     tgt,tgt_labels = target_data.to("cuda") ,target_label.to("cuda")
    
#     D_src = torch.ones(len(src), 1).to("cuda") # Discriminator Label to real
#     D_tgt = torch.zeros(len(tgt), 1).to("cuda") # Discriminator Label to fake
#     D_labels = torch.cat([D_src, D_tgt], dim=0)
#     input_data = torch.cat([src, tgt], dim=0)
#     input_labels = torch.cat([src_labels, tgt_labels], dim=0)

#     output = F(input_data)
#     # output,_ = C(F(input_data))

#     latent_embedded = TSNE(n_components=2).fit_transform(output.cpu().detach().numpy())
#     N = 10
#     X,Y = latent_embedded[:, 0], latent_embedded[:, 1] 

#     # print(latent_embedded[:, 0].shape , target.shape)
#     # for i , j , l in zip(latent_embedded[:, 0], latent_embedded[:, 1] , input_labels.cpu().numpy()):
#     for i , j , l in zip(latent_embedded[:, 0], latent_embedded[:, 1] , D_labels.cpu().numpy()):
#         # plt.figure(figsize=(8, 6))
#         if l == 1:
#             c= "r"
#         else:
#             c = "b"
#         # c = cm.rainbow(int(255 * l / 10))
#         # plt.text(i,j, str(l),color=c,fontdict={'weight': 'bold', 'size': 9}) #在指定位置放置文本

#     # print(X.min(), X.max(),Y.min(), Y.max())
#         plt.scatter(i,j, c=c, marker='o', edgecolor='none')
        
#     if count == 15:  
#         plt.xlim(-15,15)
#         plt.ylim(-15,15)
#         # plt.xlim(X.min(), X.max())
#         # plt.ylim(Y.min(), Y.max())
#         plt.show()

#         count=0
#     count+=1



