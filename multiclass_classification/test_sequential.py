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

import csv
import random
import numpy as np
import pandas as pd
from utils.mydataset import TestDataset2
from models.resnet import ResNet18
from models.vgg16 import *
from models.sliceRNN import *
from models.neck import *

import progressbar
def parse_args():
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument('--modelpath', default="model.pt", type=str, help='model path')
    parser.add_argument('--inputsize', default=224, type=int, help='inputsize') 
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    test_dataset =TestDataset2(size = args.inputsize)
    model = NeckRNN(inputsize=args.inputsize)
    
    model.load_state_dict(torch.load(args.modelpath,map_location='cpu'))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model.cuda()
    ans=[]
    model.eval()
    loader = DataLoader(
            test_dataset, 
            batch_size=1,
            num_workers=0,
            shuffle=False
            )
    print(len(loader))
    bar = progressbar.ProgressBar(maxval=len(loader), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    for i,(image, label1,label2) in enumerate(loader):
        # print(label2)
        image=torch.squeeze(image)
        image=image.unsqueeze(1)
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            image=image.cuda()
        outputs=model(image)
        bar.update(i+1)
            
        
        outputs=(outputs.detach().cpu().numpy())
        outputs=np.where(outputs>0,1,0)
        for j in range(image.size()[0]):
            ans.append((label1[j][0],label2[j][0],outputs[j]))
        del image
        
    bar.finish()
            
    # print(ans)
    outfile=open("out.csv","w")
    outfile.write("dirname,ID,ich,ivh,sah,sdh,edh\n")
    for label1,label2,output in ans:
        outfile.write(label1+","+label2)
        for i in range(5):
            outfile.write(","+str(output[i]))
    
        outfile.write("\n")
    del model 
    