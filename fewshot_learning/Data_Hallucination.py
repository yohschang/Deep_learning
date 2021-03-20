# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 16:37:13 2020

@author: YX
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 20:15:24 2020
ref: https://github.com/Hsankesara/Prototypical-Networks/blob/master/prototypical-net.ipynb
@author: YX
"""

import os
import sys
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd
from trainset_episode import get_episode

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

# fix random seeds for reproducibility
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        # print(path)
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

class GeneratorSampler(Sampler):
    def __init__(self, sampled_sequence,mode = 2):
        if mode == 1:
            episode_df = pd.read_csv(sampled_sequence).set_index("episode_id")
            self.sampled_sequence = episode_df.values.flatten().tolist()
        else:
            self.sampled_sequence = sampled_sequence

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)


#%%
def conv_block(in_channel , out_channel):
    bn = nn.BatchNorm2d(out_channel)
    return nn.Sequential(
        nn.Conv2d(in_channel , out_channel ,3 ,padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
        )



class convnet(nn.Module):
    def __init__(self , in_channel = 3 , hid_channel = 64 , out_channel = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channel , hid_channel),
            conv_block(hid_channel , hid_channel),
            conv_block(hid_channel , hid_channel),
            conv_block(hid_channel , hid_channel),
            )
    def forward(self,x):
        x = self.encoder(x)
       
        return x.view(x.size(0),-1)
    
class generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1600*2,2400)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout()
        self.linear2 = nn.Linear(2400,2400)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout()
        self.linear3 = nn.Linear(2400,1600)
        self.relu3 = nn.ReLU()

    def forward(self,x_in,M,seed):
        H_stack = torch.tensor([]).to("cuda")
        for _ in range(M):
            if seed :
                torch.manual_seed(1234)
                noise = torch.randn(x_in.size()).to("cuda")
            else :
                noise = torch.randn(x_in.size()).to("cuda")
            x = torch.cat((x_in,noise)) 
            x = self.drop1(self.relu1(self.linear1(x)))
            x = self.drop2(self.relu2(self.linear2(x)))
            x = self.relu3(self.linear3(x))
            H_stack = torch.cat((H_stack , x)) 
        return H_stack.view(-1,1600)
        # return torch.tensor([]).to("cuda")

model = convnet().to("cuda")
generator = generator().to("cuda")
# print(generator)


ways = 5
N_shot = 5 # per way
query = 50 # per way
max_episode = 800
M = 100 # hallucination

#%%
# train_csv = r"C:\Users\YH\Desktop\CVDL\HW4\hw4_data\hw4_data\train.csv"
# train_data_dir = r"C:\Users\YH\Desktop\CVDL\HW4\hw4_data\hw4_data\train"
# train_dataset = MiniDataset(train_csv, train_data_dir)
# data_stack , label_stack = get_episode(ways ,N_shot, query, max_episode)
# train_loader = DataLoader(train_dataset, batch_size= ways * (query + N_shot),
#     sampler=GeneratorSampler(data_stack))


optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
gen_optimizer = torch.optim.Adam(generator.parameters(), lr = 0.001)
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001)
# gen_optimizer = torch.optim.SGD(generator.parameters(), lr = 0.0001)

def get_centroid(model , S_cls, shots):
    return torch.sum(model((S_cls)), 0).unsqueeze(1).transpose(0,1) / shots


def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    n_x = x.shape[0]
    n_y = y.shape[0]
    # print(x.unsqueeze(1).expand(n_x, n_y, -1).size() , y.unsqueeze(0).expand(n_x, n_y, -1).size())
    # print(x)
    # print(y)

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)  #calculate each embadding query with 5 ways
        return distances  # each row is the distence sum of each query with different class
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)



def train(model,generator) :
    max_test_acc = 0
    max_epoch = 30
    for epoch in range(max_epoch):
        total_loss = 0
        total_acc = 0
        model = model.train()
        generator = generator.train()
        for (data, target) in tqdm(train_loader):
            optimizer.zero_grad()
            gen_optimizer.zero_grad()
            
            data = data.to("cuda")
            label_encoder = {target[i * N_shot] : i for i in range(ways)}
            # print(label_encoder)
            query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[ways * N_shot:]])
            
            output = model(data)
            
            support_output = output[:ways * N_shot]
    
            rand_sample = np.random.randint(0,N_shot,(ways,)) + np.array([i for i in range(0,N_shot*ways , N_shot)])
            new_support = torch.tensor([]).to("cuda")
            for index , r in enumerate(rand_sample):
                hallucinate = generator(support_output[r,...] , M , False)
                hallucinate = torch.cat((support_output[index*N_shot:(index+1)*N_shot,...] , hallucinate))
                new_support = torch.cat((new_support,hallucinate) , dim = 0)

            query_output = output[ways * N_shot:]
            # print(new_support.size())
            
            prototypes = new_support.reshape(ways, N_shot + M, -1).mean(dim=1)  ##central of each class
            # print(prototypes.size() , query_output.size())
            distances = pairwise_distances(query_output, prototypes, "dot")
            log_p_y = (-distances).log_softmax(dim=1)
            loss = F.cross_entropy(log_p_y, query_label)
            
            loss.backward()
            optimizer.step()
            gen_optimizer.step()
            
            y_pred = (-distances).softmax(dim=1)
            acc = torch.mean((torch.argmax(log_p_y, 1) == query_label).float())
            total_acc+=acc.item()
            total_loss += loss.item()
            # print(acc.item())
        
        print("LOSS: " + str(round(total_loss / max_episode , 5)))
        print("ACC: " + str(round(total_acc / max_episode , 8)))
        test_acc = test(model,generator)
        if test_acc > max_test_acc:
            save_path = r"hw4_2_model.pth"
            state = {"cnn_state_dict" : model.state_dict(), 
                      "generator_state_dict" : generator.state_dict(),
                      "gen_optimizer" : gen_optimizer.state_dict(),
                      "optimizer":optimizer.state_dict()}
            torch.save(state,save_path)
            max_test_acc = test_acc
        print(max_test_acc)

            
    
    
#%%
def test(model,generator,save_path):
    
    head = ["episode_id"]
    for i in range(75):
        head.append("query"+str(i))
    df = pd.DataFrame(columns=head)
    
    
    N_query =15
    N_shot_test= 1
    N_way = 5
    
    total_loss = 0
    total_acc = 0

    model = model.eval()
    generator = generator.eval()
    count = 0
    with torch.no_grad():
        for (data, target) in tqdm(test_loader):
            optimizer.zero_grad()
            gen_optimizer.zero_grad()
            
            data = data.to("cuda")
            label_encoder = {target[i * N_shot_test] : i for i in range(N_way)}
            query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[N_way * N_shot_test:]])
            
            output = model(data)
            
            support_output = output[:N_way * N_shot_test]
    
            np.random.seed(1234) 
            rand_sample = np.random.randint(0,N_shot_test,(N_way,)) + np.array([i for i in range(0,N_shot_test*N_way , N_shot_test)])
            new_support = torch.tensor([]).to("cuda")
            for index , r in enumerate(rand_sample):
                hallucinate = generator(support_output[r,...] , M , True)
                hallucinate = torch.cat((support_output[index*N_shot_test:(index+1)*N_shot_test,...] , hallucinate))
                new_support = torch.cat((new_support,hallucinate) , dim = 0)

            query_output = output[N_way * N_shot_test:]
            prototypes = new_support.reshape(N_way, N_shot_test + M, -1).mean(dim=1)  ##central of each class
            distances = pairwise_distances(query_output, prototypes, "dot")  
            log_p_y = (-distances).log_softmax(dim=1)
            y_pred = (-distances).softmax(dim=1)
            acc = torch.mean((torch.argmax(log_p_y, 1) == query_label).float())
            total_acc+=acc.item()
    
            dataset = np.insert(torch.argmax(log_p_y, 1).cpu().numpy(),0,count)
            df.loc[count] = dataset
            count+=1
        
        df.to_csv(save_path,index = 0)
        print("===================test=======================")
        print("ACC: " + str(round(total_acc / count , 5)))
        print("===================test=======================")
        return round(total_acc / count , 5)
        

#%%

import sys
test_csv  = sys.argv[1]
test_data_dir = sys.argv[2]
testcase_csv = sys.argv[3]
save_path = sys.argv[4]

# test_csv = r"C:\Users\YH\Desktop\CVDL\HW4\hw4_data\hw4_data\val.csv"
# test_data_dir = r"C:\Users\YH\Desktop\CVDL\HW4\hw4_data\hw4_data\val"
# testcase_csv = r"C:\Users\YH\Desktop\CVDL\HW4\hw4_data\hw4_data\val_testcase.csv"
N_query =15
N_shot_test= 1
N_way = 5

test_dataset = MiniDataset(test_csv, test_data_dir)

test_loader = DataLoader(
    test_dataset, batch_size=N_way * (N_query + N_shot_test),
    sampler=GeneratorSampler(testcase_csv,mode = 1))

if os.path.exists(r"hw4_2_model.pth"):
    checkpoint = torch.load(r"hw4_2_model.pth")
    model.load_state_dict(checkpoint['cnn_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
    
  

# train(model,generator)
test(model,generator,save_path)


#%%
# save_path = r"hw4_2_model.pth"
# state = {"cnn_state_dict" : model.state_dict(), 
#           "generator_state_dict" : generator.state_dict(),
#           "gen_optimizer" : gen_optimizer.state_dict(),
#           "optimizer":optimizer.state_dict()}
# torch.save(state,save_path)


#%%TSNE

# ways = 5
# N_shot = 80 # per way
# M = 20 # hallucination

# train_csv = r"C:\Users\YH\Desktop\CVDL\HW4\hw4_data\hw4_data\train.csv"
# train_data_dir = r"C:\Users\YH\Desktop\CVDL\HW4\hw4_data\hw4_data\train"
# train_dataset = MiniDataset(train_csv, train_data_dir)
# data_stack , label_stack = get_episode(ways ,N_shot, query, max_episode)
# train_loader = DataLoader(train_dataset, batch_size= ways * (query + N_shot),
#     sampler=GeneratorSampler(data_stack))

# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from matplotlib import cm

# def tsne(model,generator):
    
#     total_loss = 0
#     total_acc = 0

#     model = model.eval()
#     generator = generator.eval()
#     count = 0
#     with torch.no_grad():
#         for (data, target) in tqdm(train_loader):
#             optimizer.zero_grad()
#             gen_optimizer.zero_grad()
            
#             data = data.to("cuda")
#             label_encoder = {target[i * N_shot] : i for i in range(ways)}
#             query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[ways * N_shot:]])
#             output = model(data)            
#             support_output = output[:ways * N_shot]   
#             rand_sample = np.random.randint(0,N_shot,(ways,)) + np.array([i for i in range(0,N_shot*ways , N_shot)])
#             new_support = torch.tensor([]).to("cuda")
#             for index , r in enumerate(rand_sample):
#                 hallucinate = generator(support_output[r,...] , M , False)
#                 hallucinate = torch.cat((support_output[index*N_shot:(index+1)*N_shot,...] , hallucinate))
#                 new_support = torch.cat((new_support,hallucinate) , dim = 0)

#             # print(new_support.size())
#             LABEL = np.arange(len(new_support))
#             latent_embedded = TSNE(n_components=2).fit_transform(new_support.cpu().detach().numpy())
#             N = 10
#             X,Y = latent_embedded[:, 0], latent_embedded[:, 1] 

#             for i , j , l in zip(latent_embedded[:, 0], latent_embedded[:, 1],LABEL):
#                 # plt.figure(figsize=(8, 6))
#                 if l//100 == 0:
#                     c= "r"
#                 elif l//100 == 1:
#                     c= "g"
#                 elif l//100 == 2:
#                     c= "b"
#                 elif l//100 == 3:
#                     c= "y"
#                 else:
#                     c = "k"
#                 # c = cm.rainbow(int(255 * l / 10))
#                 # plt.text(i,j, str(l),color=c,fontdict={'weight': 'bold', 'size': 9}) #在指定位置放置文本
        
#             # print(X.min(), X.max(),Y.min(), Y.max())
#                 if l %100 < 80:
#                     plt.scatter(i,j, c=c, marker="x", edgecolor='none')
#                 else:
#                     plt.scatter(i,j, c=c, marker="^", edgecolor='none')
#                     plt.xlim(X.min(), X.max())
#                     plt.ylim(Y.min(), Y.max())
#             plt.show()
#             # if count == 15:  
#             #     plt.xlim(-15,15)
#             #     plt.ylim(-15,15)
 
#             #     count=0
#             # count+=1
# if os.path.exists(r"hw4_2_model.pth"):
#     checkpoint = torch.load(r"hw4_2_model.pth")
#     model.load_state_dict(checkpoint['cnn_state_dict'])
#     generator.load_state_dict(checkpoint['generator_state_dict'])
#     # optimizer.load_state_dict(checkpoint['optimizer'])
#     # gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
    
  

# tsne(model,generator)


