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
from utils.mydataset import *
from utils.evaluator import eval2

from utils.aslloss import AsymmetricLossOptimized
from models.sliceRNN import *
from models.vgg16 import *
from models.neck import *


def parse_args():
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate (default: 0.0001)')
    parser.add_argument('--bs', default=4, type=int, help='batch size(default: 48)') 
    parser.add_argument('--modelpath', default="model.pt", type=str, help='model path') 
    
    parser.add_argument('--pretrainmodelpath', default="premodelvgg_71.pt", type=str, help='model path') 
    parser.add_argument('--epochs', default=20, type=int, help='epoch num') 
    parser.add_argument('--inputsize', default=224, type=int, help='inputsize') 
    parser.add_argument('--loaded', default="f", type=str, help='model loaded?t,f)') 
    
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    train_dataset = Dataset_aug_patient(size=args.inputsize, settype='train')
    val_dataset = Dataset_aug_patient(size=args.inputsize, settype='valid')
    print("trainset num = %d\tvalidddet num = %d"%(len(train_dataset),len(val_dataset)))
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,
        num_workers=0,
        shuffle=True
        )
    FE=0
    if args.loaded=='p':
        pretrainmodel=VGG16(inputsize=args.inputsize)
        pretrainmodel.load_state_dict(torch.load(args.pretrainmodelpath,map_location='cpu'))
        FE=pretrainmodel.extractor
        model = NeckRNN(inputsize = args.inputsize,FE=FE)
    else:
        model = NeckRNN(inputsize = args.inputsize)
       
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model.cuda()
    if args.loaded=='t':
        model.load_state_dict(torch.load(args.modelpath,map_location='cpu'))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = AsymmetricLossOptimized()
    for epoch in range(args.epochs):
        print("epoch %d:"%(epoch))
        model.train()
        
        for batchnum, (image, label) in enumerate(train_loader):
            
            image=torch.squeeze(image)
            image=image.unsqueeze(1)
            label=torch.squeeze(label)
            if torch.cuda.is_available():
                image=image.cuda()
                label=label.cuda()
            optimizer.zero_grad()
            outputs=model(image)
            
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
        
            # if batchnum%10==0:
            #     print("batchnum=%d,loss=%.4f"%(batchnum,loss.item()))

                
            del image
            del label
            
        torch.save(model.state_dict(),str(epoch)+args.modelpath)
        
        print("validset:")
        eval2(model,val_dataset)
        print("trainset:")
        eval2(model,train_dataset)
    
    del model 
    