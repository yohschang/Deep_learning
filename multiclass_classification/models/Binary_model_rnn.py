'''
http://cs231n.stanford.edu/reports/2017/pdfs/903.pdf
https://github.com/czanoci/cs231n_cris_jim
'''

import torch
import torch.nn as nn
from torch.nn import init
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import sys
from models.vgg16 import *
# from vgg16 import *

class Sliceconv(nn.Module): 
    def __init__(self,inputsize=512,hiddensize=512,outputsize=512):
        super(Sliceconv, self).__init__()
        self.conv=torch.nn.Conv1d(inputsize,outputsize,kernel_size=5,stride = 1 ,padding=2)
        
        # self.conv=nn.Sequential(
        #     #torch.nn.Conv1d(inputsize,hiddensize,kernel_size=5,stride = 1 ,padding=2),
        #     torch.nn.Conv1d(hiddensize,outputsize,kernel_size=5,stride = 1 ,padding=2)
        # )
    def forward(self, x):
        x=x.permute(1, 0)
        x=x.unsqueeze(0)
        x=self.conv(x)
        x=x.squeeze(0)
        x=x.permute(1, 0)
        return x

#%%
class SliceRNN(nn.Module):
    def __init__(self,inputsize=512,hiddensize=64,layer = 4,rnntype='GRU'):
        super(SliceRNN, self).__init__()
        # self.conv1 = Sliceconv()
        self.type=rnntype
        # self.outputsize=outputsize
        self.layer = layer 
        self.hiddensize=hiddensize
        
        if self.type=='GRU':
            self.RNN=nn.GRU(inputsize,self.hiddensize,layer,batch_first=True,bidirectional=True)
        elif self.type=='LSTM':
            self.RNN=nn.LSTM(inputsize,self.hiddensize,layer,batch_first=True,bidirectional=True)
        else:
            print("RNN type ERROR!!!!")
            qwe
        # self.linear=nn.Linear(hiddensize*2,outputsize)
    def forward(self, x):
        # x = self.conv1(x)
        x=x.unsqueeze(0)
        if self.type=='GRU':
            output,hn=self.RNN(x)
        elif self.type=='LSTM':
            output,hn=self.RNN(x)
            
        output=torch.squeeze(output,0)
        # output=self.linear(output)
        return output
    
#%%
class NeckRNN(nn.Module):
    def __init__(self,inputsize=224,hiddensize=256,outputsize=5,layer = 4,rnntype='GRU',FE=None):
        super(NeckRNN, self).__init__()
        if FE==None:
            self.cnn=VGG16_FE(inputsize=inputsize)
        else:
            self.cnn=FE
        self.rnn=SliceRNN(hiddensize=hiddensize,layer = layer,rnntype=rnntype)
        # self.linear=nn.Linear(512,outputsize)
        
    def forward(self, x):
        x=self.cnn(x)
        print(x.size())
        y=self.rnn(x)
        x=x+y
        # x=self.linear(F.relu(x))
        return x
#%% 
class DecoderBinaryRNN(nn.Module):
    def __init__(self, hidden_size, cnn_output_size, num_labels,vgg = None,mode = "gru"):
        """Set the hyper-parameters and build the layers."""
        super(DecoderBinaryRNN, self).__init__()
    
        self.mode = mode
        self.num_labels = num_labels
        self.SRNN = SliceRNN()
        # self.conv1 = Sliceconv()
        # self.NeckRNN = NeckRNN(FE=vgg)
        self.linear_img_to_lstm = nn.Linear(cnn_output_size, hidden_size)
        if self.mode == "lstm":
            self.lstm = nn.LSTM(1, hidden_size, 1, batch_first=True, bidirectional=True)
            self.linear_final = nn.Linear(hidden_size*2, 1)
        elif self.mode == "gru":
            self.gru = nn.GRU(1, hidden_size, 1, batch_first=True , bidirectional=True)
            self.linear_final = nn.Linear(hidden_size*2, 1)

    def forward(self, cnn_features):
        
        # cnn_features = self.conv1(cnn_features)
        rnn_features = self.SRNN(cnn_features)
        # rnn_features = self.NeckRNN(cnn_features)
        # print(rnn_features.size())
        h0 = torch.unsqueeze(self.linear_img_to_lstm(rnn_features), 0).to("cuda")
        c0 = torch.autograd.Variable(torch.zeros(h0.size(0), h0.size(1), h0.size(2)), requires_grad = False).to("cuda")
        zero_input = torch.autograd.Variable(torch.zeros(cnn_features.size(0), self.num_labels, 1), requires_grad = False).to("cuda")

        if self.mode == "lstm":
            hiddens, _ = self.lstm(zero_input, (h0.repeat(2,1,1), c0.repeat(2,1,1)))
        elif self.mode == "gru":
            hiddens, _ = self.gru(zero_input, h0.repeat(2,1,1))

        unbound = torch.unbind(hiddens, 1)
        combined = [self.linear_final(elem) for elem in unbound]
        combined = torch.stack(combined, 1).squeeze(2)
        
        return combined
    
#%%

if __name__ == "__main__":
    hidden_size = 256
    cnn_output_size = 128
    num_labels = 5
    # birnn = SliceRNN().to("cuda")   
    birnn = DecoderBinaryRNN(hidden_size, cnn_output_size, num_labels,mode = "gru").to("cuda")   
    # birnn = NeckRNN().to("cuda")   
    in_p = torch.rand(2,512).to("cuda")  
    # in_p = torch.rand(2,1,224,224).to("cuda")  
    
    out = birnn(in_p)
    print(out.size() )
    
    # print(out.size())




