import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.vgg16 import *
from models.lrnn_srnn import *
import cv2 as cv
from PIL import Image
from to_kaggle import to_kaggle
filenameToPILImage = lambda x: Image.open(x)


class TestDataset_aug(Dataset):
    def __init__(self, data_dir,size = 32):
        patients=os.listdir(data_dir)
        self.size = size
        self.v=[]
        for f in patients:
            arr=[]
            for path in os.listdir(os.path.join(data_dir,f)):
                arr.append((os.path.join(data_dir,f,path),f,path))
            arr.sort(key=lambda x: int(x[2].split('.')[0].split('_')[-1]))
            self.v.append(arr)
            
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation((0,360)),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
            # transforms.RandomAffine(degrees=180,scale=(0.8,1.4))
            ])
            
        self.read_img = transforms.Compose([
            filenameToPILImage,
            transforms.Resize((size,size), interpolation=2),
            ])
        self.clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def augmentation(self , image_stack,seed):
        for idx , img in enumerate(image_stack):
            img = np.array(img) 
            img = self.clahe.apply(img)
            image_stack[idx,:,:] = img
        return(image_stack)

    def __getitem__(self, index):
        paths = [filepath for filepath,_,_ in self.v[index]]
        dirnames = [dirname for _,dirname,_ in self.v[index]]
        IDs = [ID for _,_,ID in self.v[index]]
        image_list = [self.read_img(path) for path in paths]
        image = np.concatenate(image_list).reshape(-1,self.size,self.size)
        seed = np.random.randint(2021)
        image = self.augmentation(image,seed).transpose(1,2,0)
        image = transforms.ToTensor()(image).float()
        
        return image, dirnames,IDs

    def __len__(self):
        return len(self.v)   

def parse_args():
    parser = argparse.ArgumentParser(description="testing")
    parser.add_argument('--data_dir' , default = "Blood_data/test" , type=str, help='datapath')
    parser.add_argument('--output_dir', default="pred.csv", type=str, help='predict file save path')     
    parser.add_argument('--to_kaggle_path', type=str, default='pred_kaggle.csv',
                    help='path of predict file in kaggle format')
    return parser.parse_args()

def test(model , testset, output_dir):   
    ans=[]
    model.eval()
    loader = DataLoader(testset,batch_size=1,num_workers=0,shuffle=False)
 
    for (image, label1,label2) in tqdm(loader):
        image=torch.squeeze(image)
        image=image.unsqueeze(1)
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            image=image.cuda()
        outputs=model(image)
        
        outputs=(outputs.detach().cpu().numpy())
        outputs=np.where(outputs>0,1,0)
        for j in range(image.size()[0]):
            ans.append((label1[j][0],label2[j][0],outputs[j]))
        del image
    outfile=open(output_dir,"w")
    outfile.write("dirname,ID,ich,ivh,sah,sdh,edh\n")
    for label1,label2,output in ans:
        outfile.write(label1+","+label2)
        for i in range(5):
            outfile.write(","+str(output[i]))
        outfile.write("\n")
    outfile.close()


if __name__=='__main__':
    args = parse_args()
    testset = TestDataset_aug(data_dir = args.data_dir , size = 224)
  
    pretrainmodel=VGG16(inputsize=224)
    FE=pretrainmodel.extractor
            
    model = nn.Sequential(
        FE,
        SRNN_LRNN()
    )

    if os.path.exists("lrnn_conv_srnn.pt"):
        checkpoint = torch.load("lrnn_conv_srnn.pt")
        model.load_state_dict(checkpoint)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model.cuda()

    test(model , testset , args.output_dir)
    
    to_kaggle(args.output_dir , args.to_kaggle_path)
