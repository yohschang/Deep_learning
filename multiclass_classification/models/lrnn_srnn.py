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


#%%
class Sliceconv(nn.Module):
    def __init__(self,inputsize=512,hiddensize=512,outputsize=5):
        super(Sliceconv, self).__init__()
        self.conv=torch.nn.Conv1d(inputsize,outputsize,kernel_size=5,stride = 1 ,padding=2)
        
        
    def forward(self, x):
        # x=x.permute(1, 0)  # 3* 512
        # x=x.unsqueeze(0)   # 512* 3
        x=self.conv(x)   #1*512 *3
        # x=x.squeeze(0)
        x=x.permute(0,2,1)
        return x
    
#%% 
class DecoderBinaryRNN(nn.Module):
    def __init__(self, hidden_size, cnn_output_size, num_labels,vgg = None,mode = "lstm"):
        """Set the hyper-parameters and build the layers."""
        super(DecoderBinaryRNN, self).__init__()
    
        self.mode = mode
        self.num_labels = num_labels
        self.linear_img_to_lstm = nn.Linear(cnn_output_size, hidden_size)
        if self.mode == "lstm":
            self.lstm = nn.LSTM(1, hidden_size, 1, batch_first=True, bidirectional=True)
            self.linear_final = nn.Linear(hidden_size*2, 1)
        elif self.mode == "gru":
            self.gru = nn.GRU(1, hidden_size, 1, batch_first=True , bidirectional=True)
            self.linear_final = nn.Linear(hidden_size*2, 1)

    def forward(self, cnn_features):

        h0 = torch.unsqueeze(self.linear_img_to_lstm(cnn_features), 0).to("cuda")
        c0 = torch.autograd.Variable(torch.zeros(h0.size(0), h0.size(1), h0.size(2)), requires_grad = False).to("cuda")
        zero_input = torch.autograd.Variable(torch.zeros(cnn_features.size(0), self.num_labels, 1), requires_grad = False).to("cuda")

        if self.mode == "lstm":
            hiddens, _ = self.lstm(zero_input, (h0.repeat(2,1,1), c0.repeat(2,1,1)))
        elif self.mode == "gru":
            hiddens, _ = self.gru(zero_input, h0.repeat(2,1,1))
        # print( hiddens.size())
        # unbound = torch.unbind(hiddens, 1)
        # # print(unbound)
        # combined = [self.linear_final(elem) for elem in unbound]
        # combined = torch.stack(combined, 1).squeeze(2)
        
        return hiddens
    
#%%
class SRNN_LRNN(nn.Module):
    def __init__(self,inputsize=512,hiddensize=512,layer = 1,outputsize = 5,rnntype='GRU'):
        super(SRNN_LRNN, self).__init__()
        
        self.lrnn = DecoderBinaryRNN(128,512,5)
        self.conv=Sliceconv(inputsize=1280,hiddensize=512,outputsize=512)
        self.type=rnntype
        self.layer = layer 
        self.hiddensize=hiddensize
        
        if self.type=='GRU':
            self.RNN=nn.GRU(inputsize,self.hiddensize,layer,batch_first=True,bidirectional=True)
        elif self.type=='LSTM':
            self.RNN=nn.LSTM(inputsize,self.hiddensize,layer,batch_first=True,bidirectional=True)
        else:
            print("RNN type ERROR!!!!")
            qwe
        self.linear=nn.Linear(hiddensize*2,outputsize)
    def forward(self, x):
        x = self.lrnn(x)
        # print(x.size())
        x = torch.reshape(x , (x.size(0),-1))
        x=x.unsqueeze(0)
        
        x = x.permute(0,2,1)
        x = self.conv(x)
        # print(x.size())
        if self.type=='GRU':
            output,hn=self.RNN(x)
        elif self.type=='LSTM':
            output,hn=self.RNN(x)
            
        output=torch.squeeze(output,0)
        output=self.linear(output)
        return output
#%%

if __name__ == "__main__":
    hidden_size = 512
    cnn_output_size = 512
    num_labels = 5
    # birnn = SRNN_LRNN().to("cuda")   
    birnn = DecoderBinaryRNN(hidden_size, cnn_output_size, num_labels,mode = "gru").to("cuda")   
    # birnn = NeckRNN().to("cuda")   
    in_p = torch.rand(3,512).to("cuda")  
    # in_p = torch.rand(2,1,224,224).to("cuda")  
    
    out = birnn(in_p)
    print(out.size() )
    
    # print(out.size())




