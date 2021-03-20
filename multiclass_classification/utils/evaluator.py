import torch

from torch.utils.data import Dataset,DataLoader
import progressbar
import numpy as np
def eval(model,dataset):
    model.eval()
    with torch.no_grad():
        loader = DataLoader(
            dataset, 
            batch_size=24,
            num_workers=0,
            shuffle=True
            )
        print("total = %d"%(len(loader)))
        TP=0
        FP=0
        FN=0
        bar = progressbar.ProgressBar(maxval=len(loader), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        
        for batchnum, (image, label) in enumerate(loader):
            bar.update(batchnum+1)
            if torch.cuda.is_available():
                image=image.cuda()
                label=label.cuda()
            
            outputs=model(image)
            
            ans = torch.stack((outputs.view(-1),label.view(-1)),1)
            for i in range(ans.size()[0]):
                if ans[i,0]>0:
                    if(ans[i,1]==1):
                        TP+=1
                    else:
                        FP+=1
                elif(ans[i,1]==1):
                    FN+=1    
            if(batchnum>300):
                break
            del image
            del label
        bar.finish()
    print("TP=%d\tFP=%d\tFN=%d\t"%(TP,FP,FN))
    print("F2_score:%.5f"%(5*TP*TP/(TP*(TP+FP)+4*TP*(TP+FN))))


def eval2(model,dataset):
    model.eval()
    with torch.no_grad():
    
        loader = DataLoader(
            dataset, 
            batch_size=1,
            num_workers=0,
            shuffle=False
            )
        TP=0
        FP=0
        FN=0
        bar = progressbar.ProgressBar(maxval=len(loader), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        
        for batchnum, (image, label) in enumerate(loader):
            bar.update(batchnum+1)
            
            if torch.cuda.is_available():
                image=image.cuda()
                label=label.cuda()
            image=torch.squeeze(image)
            image=image.unsqueeze(1)
            label=torch.squeeze(label)
            
            outputs=model(image)
            ans = torch.stack((outputs.flatten(),label.flatten()),1)
            for i in range(ans.size()[0]):
                if ans[i,0]>0:
                    if(ans[i,1]==1):
                        TP+=1
                    else:
                        FP+=1
                elif(ans[i,1]==1):
                    FN+=1    
            
            del image
            del label
            if batchnum>400:break
        bar.finish()
        print("TP=%d\tFP=%d\tFN=%d\t"%(TP,FP,FN))
        print("F2_score:%.5f"%(5*TP*TP/(TP*(TP+FP)+4*TP*(TP+FN))))
        

   