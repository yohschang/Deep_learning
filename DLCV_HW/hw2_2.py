# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 18:41:16 2020

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
from torch.utils import model_zoo
from PIL import Image
import numpy as np
from scipy import misc
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import time
import random
import matplotlib.pyplot as plt
import matplotlib 

import warnings
warnings.filterwarnings("ignore")

def read_masks(filepath):
    '''
    Read masks from directory and tranform to categorical
    '''

    masks = np.empty((512,512))
    mask = misc.imread(filepath,mode = "RGB")
    mask = (mask >= 128).astype(int)
    mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
    masks[mask == 3] = 0  # (Cyan: 011) Urban land 
    masks[mask == 6] = 1  # (Yellow: 110) Agriculture land 
    masks[mask == 5] = 2  # (Purple: 101) Rangeland 
    masks[mask == 2] = 3  # (Green: 010) Forest land 
    masks[mask == 1] = 4  # (Blue: 001) Water 
    masks[mask == 7] = 5  # (White: 111) Barren land 
    masks[mask == 0] = 6  # (Black: 000) Unknown
    masks[mask == 4] = 6  # (Red: 100) Unknown
    return masks

def return_mask(mask):
    maskimg = np.zeros((512,512,3))
    maskimg[mask == 1 ,0] = 255
    maskimg[mask == 2 ,0] = 255
    maskimg[mask == 5 ,0] = 255
    
    maskimg[mask == 0 ,1] = 255
    maskimg[mask == 1 ,1] = 255
    maskimg[mask == 3 ,1] = 255
    maskimg[mask == 5 ,1] = 255
    
    maskimg[mask == 0 ,2] = 255
    maskimg[mask == 2 ,2] = 255
    maskimg[mask == 4 ,2] = 255
    maskimg[mask == 5 ,2] = 255
    return maskimg

class hwdata(Dataset):
    def __init__(self , root , transform = None):
        self.root = root
        self.transform = transform
        self.images = None
        self.labels = None
        self.sat_filenames = []
        self.mask_filenames = []
        for filename in sorted(glob.glob(self.root + r"\*.jpg" ), key=os.path.getmtime):
            self.sat_filenames.append(filename)
            m_filename = filename.replace("sat.jpg","mask.png")
            self.mask_filenames.append(m_filename)

        self.length = len(self.sat_filenames)
        # print(self.length)
        # print(self.length)
    
    def _transform(self, img, lbl):
        
        rand_flip = np.random.rand()
        if rand_flip < 0.3:
            img = np.fliplr(img)
            lbl = np.fliplr(lbl) 
        elif rand_flip > 0.7:
            img = np.flipud(img)
            lbl = np.flipud(lbl) 
        img = transforms.ToTensor()(img.copy()).float()
        lbl = transforms.ToTensor()(lbl.copy()).long()
        # img = transforms.ColorJitter(brightness=0.5, contrast=1)(img)
        return img, lbl   
    
    def __getitem__(self,index):
        sat_image_path = self.sat_filenames[index]
        mask_image_path = self.mask_filenames[index]
        sat_image = misc.imread(sat_image_path,mode = "RGB").astype(float)
        mask_image = read_masks(mask_image_path)

        if self.transform:     #convert PIL to tensor or resize.....
            return self._transform(sat_image, mask_image)
        else:
            sat_image = transforms.ToTensor()(sat_image).float()
            mask_image = transforms.ToTensor()(mask_image)
        return sat_image , mask_image
    
    def __len__(self):
        return self.length


trainset_path = r"C:\Users\YH\Desktop\CVDL\CVDL_HW2\hw2-yohschang\hw2_data\p2_data\train"
trainset = hwdata(root = trainset_path , transform = False) 

valset_path = r"C:\Users\YH\Desktop\CVDL\CVDL_HW2\hw2-yohschang\hw2_data\p2_data\validation"
valset = hwdata(root = valset_path , transform =  False)

trainset_loader = DataLoader(trainset , batch_size = 8 ,shuffle = True)
valset_loader = DataLoader(valset , batch_size=10 , shuffle = False)


#%%
class VGG(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG, self).__init__()
        pretrained_model = models.vgg16(pretrained=pretrained)
        pretrained_params = pretrained_model.state_dict()
        keys = list(pretrained_params.keys())
        new_dict = {}
        for index, key in enumerate(self.state_dict().keys()):
            new_dict[key] = pretrained_params[keys[index]]
        self.load_state_dict(new_dict)
        # conv1 1/2
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv2 1/4
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv3 1/8
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv4 1/16
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv5 1/32
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # load pretrained params from torchvision.models.vgg16(pretrained=True)


    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = self.relu1_2(self.conv1_2(x))
        x = self.pool1(x)
        pool1 = x

        x = self.relu2_1(self.conv2_1(x))
        x = self.relu2_2(self.conv2_2(x))
        x = self.pool2(x)
        pool2 = x

        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.relu3_3(self.conv3_3(x))
        x = self.pool3(x)
        pool3 = x

        x = self.relu4_1(self.conv4_1(x))
        x = self.relu4_2(self.conv4_2(x))
        x = self.relu4_3(self.conv4_3(x))
        x = self.pool4(x)
        pool4 = x

        x = self.relu5_1(self.conv5_1(x))
        x = self.relu5_2(self.conv5_2(x))
        x = self.relu5_3(self.conv5_3(x))
        x = self.pool5(x)
        pool5 = x

        return pool1, pool2, pool3, pool4, pool5
class VGG16_fcn32(nn.Module):
    def __init__(self, backbone="vgg"):
        super().__init__()
        self.num_classes = 7
        if backbone == "vgg":
            self.features = VGG()

        # deconv1 1/16
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU()

        # deconv1 1/8
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()

        # deconv1 1/4
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        # deconv1 1/2
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        # deconv1 1/1
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()

        self.classifier = nn.Conv2d(32, 7, kernel_size=1)

    def forward(self, x):
        features = self.features(x)

        y = self.bn1(self.relu1(self.deconv1(features[4])) + features[3])

        y = self.bn2(self.relu2(self.deconv2(y)) + features[2])

        y = self.bn3(self.relu3(self.deconv3(y)) + features[1])

        y = self.bn4(self.relu4(self.deconv4(y)) + features[0])

        y = self.bn5(self.relu5(self.deconv5(y)))

        y = self.classifier(y)
        return y
model = VGG16_fcn32().to("cuda")
# print(model)

#%%
def mean_iou_score(pred, labels):
    mean_iou = 0
    ious = []
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        # print(tp_fp , tp_fn,  tp)
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        ious.append(iou)
    #     print('class #%d : %1.5f'%(i, iou))
    # print('\nmean_iou: %f\n' % mean_iou)
    return np.nanmean(ious)

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr

#%%
criterion = nn.CrossEntropyLoss().to("cuda")
optimizer = optim.Adam(model.parameters() , lr = 0.005) # momentum : how much the renewal depend on previous

def pixel_acc(pred, target):
    correct = np.sum(pred == target)
    total   = np.sum(target == target)
    return correct / total

def train(model , epoch, start_epoch):
    save_cri = 0
    for e in range(start_epoch+1 , epoch):
        if e > 0 and e % 10 == 0:
            set_learning_rate(optimizer ,0.7)
        train_acc = 0
        train_loss = 0
        mean_iou = 0
        p_acc = 0
        model = model.train()
        for batch_idx , (data , target )in enumerate(trainset_loader):
            data , target = data.to("cuda") , target.to("cuda", dtype=torch.int64)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output , target[:,0,:,:])
            loss.backward()
            optimizer.step() 

            label_pred = output.data.max(1)[1].data.cpu().numpy() #max(1) : find max in axis = 1 ; [1] after max turn the value to position
            label_true = target.data.cpu().numpy()
            train_loss += float(loss.item())
            for pred , labels in zip(label_pred , label_true):
                labels = labels[0,:,:]
                p_acc += pixel_acc(pred, labels)
                mean_iou += mean_iou_score(pred, labels)
                 
                
            print(" load_size = " +str(round(100.*batch_idx*len(data)/len(trainset_loader.dataset)))
                    + "% , loss = " + str(round(loss.item(),5)))    
        print(train_loss/ len(trainset_loader.dataset))
        train_accuracy = 100. * mean_iou / len(trainset_loader.dataset)
        P_ACC= 100.* p_acc / len(trainset_loader.dataset)
        print("m_iou : " +str(train_accuracy) +"pix_acc : "+str(P_ACC))
        val_acc = validation(model ,e)
        
        # if val_acc > save_cri:
        #     save_cri = val_acc
        #     state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': e}
        #     torch.save(state, r"C:\Users\YH\Desktop\CVDL_HW2\hw2-yohschang\hw2_2_model.pth")

#%%
def validation(model ,output_dir ):
    model = model.eval()  # turn model into evaluation mode 
    tot_loss = 0
    correct = 0
    p_acc = 0
    count = 0
    label_trues, label_preds = [], []
    with torch.no_grad():   # free gpu memory use for back-up
        for data , target in testset_loader:
            data , target = data.to("cuda") , target.to("cuda", dtype=torch.int64)
            output = model(data)
            label_pred = output.data.max(1)[1].data.cpu().numpy()
            label_true = target.data.cpu().numpy()
            for pred , labels in zip(label_pred , label_true):
                labels = labels[0,:,:].astype(int)
                p_acc += pixel_acc(pred, labels)
                mean_iou = mean_iou_score(pred, labels)

                correct += mean_iou
                predimg = return_mask(pred)           
                misc.imsave(output_dir +"\\"+str(count).zfill(4)+".png",predimg)
                # misc.imsave(output_dir+"\\"+str(count).zfill(4)+".png",labelimg)
                count+=1
    
    tot_loss /= len(testset_loader.dataset)
    P_ACC= 100.* p_acc / len(testset_loader.dataset)
    
    print("++++++++++++++++++++++++++++++++")
    import mean_iou_evaluate as mie
    pred = mie.read_masks(r"C:\Users\YH\Desktop\CVDL_HW2\hw2-yohschang\hw2_data\p2_data\pred")
    labels = mie.read_masks(r"C:\Users\YH\Desktop\CVDL_HW2\hw2-yohschang\hw2_data\p2_data\validation")
    m_iou = mie.mean_iou_score(pred, labels)
    print(" loss : " + str(tot_loss) + " Accuracy : "+ str(round(m_iou*100,5)) )
    print("pixel accuracy : " + str(P_ACC))
    print("++++++++++++++++++++++++++++++++")
    return m_iou*100


#%%
# import os
# if os.path.exists(r"C:\Users\YH\Desktop\CVDL_HW2\hw2-yohschang\hw2_2_model.pth"):
#     checkpoint = torch.load(r"C:\Users\YH\Desktop\CVDL_HW2\hw2-yohschang\hw2_2_model.pth")
#     model.load_state_dict(checkpoint['model'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     start_epoch = checkpoint['epoch']
#     # start_epoch = 0
#     print(checkpoint['epoch'])
# else:
#     start_epoch = 0
    
train(model , 120 , 0)
# torch.cuda.empty_cache()
    
#%%   
# import sys
# test_img_dir = sys.argv[1]
# output_img_dir = sys.argv[2]

# # test_img_dir = r"hw2_data\p2_data\validation"
# # output_img_dir = r"hw2_data\p2_data\pred"

# optimizer = optim.Adam(model.parameters() , lr = 0.005) # momentum : how much the renewal depend on previous

# import os
# if os.path.exists(r"hw2_2_model.pth"):
#     checkpoint = torch.load(r"hw2_2_model.pth")
#     model.load_state_dict(checkpoint['model'])
#     optimizer.load_state_dict(checkpoint['optimizer'])

# else:
#     print("please click 'hw2_2_model_download' to download problem 2 model")


# testset_path = test_img_dir 
# testset = gettestdata(root = testset_path , transform = None)
# testset_loader = DataLoader(testset , batch_size=10 , shuffle = False)
# test(model ,output_img_dir)
