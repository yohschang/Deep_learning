import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.vgg16 import *
from models.sliceRNN import *
class NeckRNN(nn.Module):
    def __init__(self,inputsize=512,hiddensize=20,outputsize=5,rnntype='GRU',FE=None):
        super(NeckRNN, self).__init__()
        if FE==None:
            self.cnn=VGG16_FE(inputsize=inputsize)
        else:
            self.cnn=FE
        self.rnn=SliceRNN(hiddensize=hiddensize,outputsize=outputsize,rnntype=rnntype)
        self.linear=nn.Linear(512+2*hiddensize,outputsize)
        
    def forward(self, x):
        x=self.cnn(x)
        y=self.rnn(x)
        x=torch.cat((x,y),1)
        x=self.linear(x)

        return x
class Neckconv(nn.Module):
    def __init__(self,inputsize=512,hiddensize=256,outputsize=5,rnntype='GRU',FE=None):
        super(Neckconv, self).__init__()
        if FE==None:
            self.cnn=VGG16_FE(inputsize=inputsize)
        else:
            self.cnn=FE
        self.rnn=Sliceconv(outputsize=512)
        self.linear=nn.Linear(512,outputsize)
        
    def forward(self, x):
        x=self.cnn(x)
        y=self.rnn(x)
        x=x+y
        x=self.linear(F.relu(x))

        return x
if __name__=="__main__":
    model = Neck()
    input = torch.randn((31,1,224,224))
    # if torch.cuda.is_available():
    #     model.cuda()
    #     input=input.cuda()

    print(input.size())
    print(model(input).size())