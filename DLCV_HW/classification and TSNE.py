# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 19:08:24 2020

@author: YX
https://www.twblogs.net/a/5c1fa04abd9eee16b3dab130
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
import time
import os
# from sklearn.manifold import TSNE
import pandas as pd


#%%create dataset
t1 = time.time()
class hwdata(Dataset):
    def __init__(self , root , transform = None):
        self.root = root
        self.transform = transform
        self.images = None
        self.labels = None
        self.filenames = []
        
        # for filename in sorted(glob.glob(self.root + r"/*.png" ), key=os.path.getmtime):
        #     label = filename.replace(root,"")[1]
        #     self.filenames.append((filename,int(label)))
         
        file_list = [file for file in os.listdir(self.root) if file.endswith('.png')]
        file_list.sort() 
        for name in file_list:
            label = name
            filename = os.path.join(self.root, name)
            self.filenames.append((filename,label))
        self.length = len(self.filenames)
        
    def _transform(self, img):   
        if np.random.rand() < 0.5:
            img   = np.fliplr(img) 
        img = transforms.ToTensor()(img.copy()).float()
        img = transforms.ColorJitter(brightness=0.5, contrast=0.5)(img)
        return img
    
    def __getitem__(self,index):
        image_path , label = self.filenames[index]
        image = Image.open(image_path)
        
        if self.transform is not None:     #convert PIL to tensor or resize.....
            image = self._transform(image)
        else:
            image = transforms.ToTensor()(image.copy()).float()
        return image , label
    
    def __len__(self):
        return self.length


#%% model
class VGG16_MOD(nn.Module):
    def __init__(self , num_class = 50):
        super().__init__()  #inherit nn.module
        model = models.vgg16(pretrained=True)
        pretrained_params = model.state_dict()
        keys = list(pretrained_params.keys())
        new_dict = {}
        for index, key in enumerate(self.state_dict().keys()):
            new_dict[key] = pretrained_params[keys[index]]
        self.load_state_dict(new_dict)
        model.classifier = nn.Sequential()  # clear origin classified layer
        self.features = model  # remain other layers

        self.fc_ = nn.Linear(512*7*7,8192)
        self.relu_ = nn.ReLU(True)
        self.dropout_ = nn.Dropout()
        self.fc0 = nn.Linear(8192,4096)
        self.relu_0 = nn.ReLU(True)
        self.dropout_0 = nn.Dropout()
        self.fc1 = nn.Linear(4096,4096)
        self.relu_1 = nn.ReLU(True)
        self.dropout_1 = nn.Dropout()
        self.fc2 = nn.Linear(4096, num_class)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_(x)
        x = self.relu_(x)
        x = self.dropout_(x)
        x = self.fc0(x)
        x = self.relu_0(x)
        x = self.dropout_0(x)
        fc1 = self.fc1(x)
        x = self.relu_1(fc1)
        x = self.dropout_1(x)
        x = self.fc2(fc1)
        return x , fc1
    
model = VGG16_MOD().to("cuda")

        
#%% train

def train(model , epoch , log_interval = 100):
    save_cri = 0
    optimizer = optim.SGD(model.parameters(), lr = 0.0005 , momentum=0.5) # momentum : how much the renewal depend on previous
    criterion = nn.CrossEntropyLoss().to("cuda")  #loss function
    model = model.train()
    iteration = 0
    improve = True
    ep = 0

    while improve:
        for batch_idx , (data , target )in enumerate(trainset_loader):
            data , target = data.to("cuda",) , target.to("cuda")
            optimizer.zero_grad()
            output ,_ = model(data)
            loss = criterion(output , target)
            loss.backward()
            optimizer.step() 
            if iteration % 100 == 0 :
                print(" load_size = " +str(round(100.*batch_idx*len(data)/len(trainset_loader.dataset))))
            iteration += 1
        accuracy = test(model)
        ep+=1
        if accuracy > save_cri:
            if accuracy > 75:
                improve = False
            save_cri = accuracy
            # save_path = r"C:\Users\YH\Desktop\CVDL_HW2\hw2-yohschang\hw2_1_model.pth"
            # state = {"state_dict" : model.state_dict(),"optimizer":optimizer.state_dict()}
            # torch.save(state,save_path)
            t2 = time.time()
            print("cost : " + str(t2-t1))
            print("epoch = " + str(ep))
        torch.cuda.empty_cache() 

    
def test(model,pred_save_pth,test_img_dir):
    critirion = nn.CrossEntropyLoss()
    model.eval()  # turn model into evaluation mode 
    test_loss = 0
    correct = 0
    pred_label = []
    image_id = []
    with torch.no_grad():   # free gpu memory use for back-up
        for data , target in testset_loader:
            data = data.to("cuda") 
            output , last_layer = model(data)
            # test_loss += critirion(output , target).item()  #sum up batch loss
            pred = output.max(1,keepdim = True)[1]  # find out which label get max possibility
            # correct += pred.eq(target.view_as(pred)).sum().item()  # sum up correct result
            
            # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            # low_dim_embs = tsne.fit_transform(last_layer.cpu().data.numpy()[:, :])
            # labels = target.cpu().numpy()[:]
            # plot_with_labels(low_dim_embs, labels)
            
            for img_id , im_pd in zip(target , pred):
                image_id.append(img_id)
                pred_label.append(int(im_pd.cpu().numpy()))
                
    dicts = {"image_id" :image_id , "label" : pred_label }
    DF = pd.DataFrame(dicts)
    DF.to_csv(pred_save_pth,index = 0)
            
    test_loss /= len(testset_loader.dataset)
    test_accuracy = 100.* correct / len(testset_loader.dataset)
    # print("_____________________________________")
    # # print("epoch: " + str(ep) )
    # print(" Accuracy : "+ str(round(test_accuracy,5)))
    # print("_____________________________________")
    return round(test_accuracy,5)

#%% load dataset
# trainset_path = r"C:\Users\YH\Desktop\CVDL_HW2\hw2-yohschang\hw2_data\p1_data\train_50"
# trainset = hwdata(root = trainset_path , transform = transforms.ToTensor()) 

# valset_path = r"C:\Users\YH\Desktop\CVDL_HW2\hw2-yohschang\hw2_data\p1_data\val_50"
# valset = hwdata(root = valset_path , transform = None)

# trainset_loader = DataLoader(trainset , batch_size=100 , shuffle = True)
# valset_loader = DataLoader(valset , batch_size=500 , shuffle = False)

# dataiter = iter(trainset_loader)
# images , labels = dataiter.next() 
    
#%% train model
# import os
# optimizer = optim.SGD(model.parameters(), lr = 0.0002 , momentum=0) # momentum : how much the renewal depend on previous

# if os.path.exists(r"C:\Users\YH\Desktop\CVDL_HW2\hw2-yohschang\hw1_model.pth"):
#     checkpoint = torch.load(r"C:\Users\YH\Desktop\CVDL_HW2\hw2-yohschang\hw1_model.pth")
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
    
# train(model ,100)

#%% test model
import sys
test_img_dir = sys.argv[1]
labelpred_file_dir = sys.argv[2]

optimizer  =  optim.SGD(model.parameters() , lr = 0.001 ) # momentum : how much the renewal depend on previous

import os
if os.path.exists(r"hw2_1_model.pth"):
    checkpoint = torch.load(r"hw2_1_model.pth")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


testset_path = test_img_dir 
labelpred_file_dir = labelpred_file_dir+"/test_pred.csv"
testset = hwdata(root = testset_path , transform = None)
testset_loader = DataLoader(testset , batch_size=500 , shuffle = False)
test(model , labelpred_file_dir ,test_img_dir)


