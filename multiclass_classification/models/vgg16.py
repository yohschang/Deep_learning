import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VGG16_FE(nn.Module):
    def __init__(self,inputsize=512):
        super(VGG16_FE, self).__init__()
        model=torchvision.models.vgg16(pretrained=True)
        self.extractor=model.features
        self.avgsize=inputsize//32
        
    def forward(self, x):
        
        x= x.expand(-1,3,-1,-1)
        x=self.extractor(x)
        x = F.avg_pool2d(x,  self.avgsize)
        x = torch.squeeze(x)
        return x

class VGG16(nn.Module):
    def __init__(self,inputsize=512,outputsize=5):
        super(VGG16, self).__init__()
        model=torchvision.models.vgg16(pretrained=True)
        self.extractor=VGG16_FE(inputsize)
        self.classifier=nn.Sequential(
            # nn.Linear(512,512),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(512,512),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(512,outputsize),            
        )

    def forward(self, x):
        x=self.extractor(x)
        x=self.classifier(x)
        return x

if __name__=="__main__":
    model = VGG16(inputsize=224)
    input = torch.randn((2,1,224,224))
    if torch.cuda.is_available():
        model.cuda()
        input=input.cuda()

    print(input.size())
    print(model(input).size())