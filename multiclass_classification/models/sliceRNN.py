import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
class SliceRNN(nn.Module):
    def __init__(self,inputsize=512,hiddensize=512,outputsize=5,rnntype='GRU'):
        super(SliceRNN, self).__init__()
        self.type=rnntype
        self.outputsize=outputsize
        self.hiddensize=hiddensize
        
        if self.type=='GRU':
            self.RNN=nn.GRU(inputsize,self.hiddensize,1,batch_first=True,bidirectional=True)
        elif self.type=='LSTM':
            self.RNN=nn.LSTM(inputsize,self.hiddensize,1,batch_first=True,bidirectional=True)
        else:
            print("RNN type ERROR!!!!")
            qwe
        self.linear=nn.Linear(hiddensize*2,outputsize)
        
        
    def forward(self, x):
        x=x.unsqueeze(0)
        if self.type=='GRU':
            h0=torch.zeros(2,1,self.hiddensize)
            if torch.cuda.is_available():
                h0=h0.cuda()
            output,hn=self.RNN(x,h0)
        elif self.type=='LSTM':
            output,hn=self.RNN(x)
        print(output.size())
        output=torch.squeeze(output,0)
        
        #output=self.linear(output)
        return output
class Sliceconv(nn.Module):
    def __init__(self,inputsize=512,hiddensize=512,outputsize=5):
        super(Sliceconv, self).__init__()
        self.conv=torch.nn.Conv1d(inputsize,outputsize,kernel_size=5,stride = 1 ,padding=2)
        
        
    def forward(self, x):
        x=x.permute(1, 0)
        x=x.unsqueeze(0)
        x=self.conv(x)
        x=x.squeeze(0)
        x=x.permute(1, 0)
        return x
class Sliceconv2(nn.Module):
    def __init__(self,inputsize=512,hiddensize=512,outputsize=5):
        super(Sliceconv2, self).__init__()
        self.conv=nn.Sequential(
            torch.nn.Conv1d(inputsize,hiddensize,kernel_size=5,stride = 1 ,padding=2),
            torch.nn.Conv1d(hiddensize,outputsize,kernel_size=5,stride = 1 ,padding=2)
        )
        
        
    def forward(self, x):
        x=x.permute(1, 0)
        x=x.unsqueeze(0)
        x=self.conv(x)
        x=x.squeeze(0)
        x=x.permute(1, 0)
        return x
class SliceRNNconv(nn.Module):
    def __init__(self,inputsize=512,hiddensize=512,outputsize=5,rnntype='GRU'):
        super(SliceRNNconv, self).__init__()
        self.type=rnntype
        self.outputsize=outputsize
        self.hiddensize=hiddensize
        
        if self.type=='GRU':
            self.RNN=nn.GRU(inputsize,self.hiddensize,1,batch_first=True,bidirectional=True)
        elif self.type=='LSTM':
            self.RNN=nn.LSTM(inputsize,self.hiddensize,1,batch_first=True,bidirectional=True)
        else:
            print("RNN type ERROR!!!!")
            qwe
        self.conv=torch.nn.Conv1d(hiddensize*2,outputsize,kernel_size=5,stride = 1 ,padding=2)
        
        
    def forward(self, x):
        x=x.unsqueeze(0)
        if self.type=='GRU':
            h0=torch.zeros(2,1,self.hiddensize)
            if torch.cuda.is_available():
                h0=h0.cuda()
            output,hn=self.RNN(x,h0)
        elif self.type=='LSTM':
            output,hn=self.RNN(x)
        
        x=output.permute(0,2,1)
        
        x=self.conv(x)
        x=x.squeeze(0)
        x=x.permute(1, 0)
        return x
if __name__=="__main__":
    model = SliceRNN(inputsize=2048,hiddensize=1024,outputsize=10,rnntype='GRU')
    input = torch.randn((1,2048))
    if torch.cuda.is_available():
        model.cuda()
        input=input.cuda()

    print(input.size())
    print(model(input).size())