import os
import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2 as cv
from PIL import Image
def dialate(x):
    x = torch.unsqueeze(torch.unsqueeze(x,0),0)
    print(x)
    kernel=torch.FloatTensor([[1],[1],[1]])
    kernel = torch.unsqueeze(torch.unsqueeze(kernel,0),0)
    print(kernel)
    x = torch.clamp(torch.nn.functional.conv2d(x, kernel, padding=(1, 0)), 0, 1)
    x = x.squeeze(0)
    x = x.squeeze(0)

    return x
if __name__=="__main__":
    x= torch.FloatTensor([[0,0,0,0,0],[0,0,0,1,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
    print(x)
    x = dialate(x)
    print(x)
    x = dialate(x)
    print(x)
    
