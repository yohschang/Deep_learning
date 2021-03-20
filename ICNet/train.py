import os
import time
import matplotlib.pyplot as plt
import datetime
import yaml
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset , DataLoader
from dataset import getdata , testdata , valdata
from models import simple_icnet
from utils import ICNetLoss, IterationPolyLR, SegmentationMetric, SetupLogger


def meaniou(pred , label):
    
    intersect = pred + label
    intersect_area = torch.sum(intersect != 0).item()
    cross_area = torch.sum(intersect == 2).item()

    if torch.sum(intersect == 2) == torch.sum(label == 1):
        iou = 1
    elif cross_area == 0 and intersect_area == 0:
        iou = 1
    else :
        iou = cross_area / intersect_area     
    return iou
    
def pixel_acc(pred, target):
    intersect = pred + target
    correct = torch.sum(pred == target).item()
    total   = torch.sum(target == target).item()
    bg = torch.sum(intersect == 0).item()
    return (correct) / (total)
    # return (correct-bg) / (total-bg)

def train(dataloader):
    epoch = 50
    max_iou = 0.72
    for e in range(epoch): 
        model.train()
        tot_pixAcc = 0
        tot_mIoU = 0
        tot_loss = 0
        count = 0
        for (images, targets) in tqdm(dataloader):  

            count+=1
            optimizer.zero_grad()
            images = images.to('cuda')
            targets = targets.to('cuda')
            
            output = model(images)
            loss = criterion(output, targets)
		
            p_out = output[0].data.max(1)[1].data
            tot_loss += loss.item()
            tot_mIoU += meaniou(p_out[0,:,:] , targets[0,0,:,:])
            tot_pixAcc += pixel_acc(p_out[0,:,:], targets[0,0,:,:])

            loss.backward()
            optimizer.step()
            
            if count%10000 == 0:
                p_out = p_out.cpu().numpy()[0,:,:]
                target = targets.cpu().numpy()[0,0,:,:]
                plt.imshow(p_out)
                plt.show()
                plt.imshow(target)
                plt.show()
                
        
        print('\n ================= train ====================')
        print('epoch : ' + str(e))
        print('tot_loss : ' + str(round(tot_loss / count , 5)))
        print('tot_mIoU : ' + str(round(tot_mIoU / count , 5)))
        print('tot_pixAcc : ' + str(round(tot_pixAcc / count , 5)))
        print('=================== train ====================')
        
        ACC = validate(model)
        
        if ACC > max_iou :
            max_iou = ACC
            save_path = r"simple_icnet_mbg.pt"
            state = {"state_dict" : model.state_dict()}
            torch.save(state,save_path)
            
            print(" \n ==== save at iou : " + str(max_iou) +" ====")
           
            
def validate(model):
    model = model.eval()
    tot_pixAcc = 0
    tot_mIoU = 0
    count = 0
    # mask_stack = torch.tensor([]).to("cuda")
    with torch.no_grad():
        for (images, targets) in tqdm(valset_loader):  
            # t1 = time.time()
            count+=1
            optimizer.zero_grad()
            images = images.to('cuda')
            targets = targets.to('cuda')
            
            output = model(images)
            p_out = output[0].data.max(1)[1].data
            
            # t2 = time.time()
            # print(t2-t1)
            

            tot_mIoU += meaniou(p_out[0,:,:] , targets[0,0,:,:])
            tot_pixAcc += pixel_acc(p_out[0,:,:], targets[0,0,:,:])
            
            # mask_stack = torch.cat((mask_stack , p_out) , dim = 0)
        print('\n ================ validate =====================')
        print('tot_mIoU : ' + str(round(tot_mIoU / count , 5)))
        print('tot_pixAcc : ' + str(round(tot_pixAcc / count , 5)))
        print('================== validate =====================')
        
        return round(tot_mIoU / count , 5)

def test(model,savepath):
    model = model.eval()
    mask_stack = torch.tensor([])
    # mask_stack = np.array([])
    count = 0
    num = 0
    with torch.no_grad():
        t1= 0
        for images in tqdm(testset_loader): 
            count+=1
            images = images.to('cuda') 
            output = model(images)
            p_out = output[0].data.max(1)[1].data.cpu().numpy().astype(np.uint8)
            # mask_stack = torch.cat((mask_stack , p_out))
            np.save(savepath+r"\mask_stack"+str(count)+".npy" , p_out[0,:,:])
            # if count == 1:
            #     mask_stack = p_out
            # else:
            #     mask_stack = np.concatenate((mask_stack , mask_stack))
            # print(mask_stack.shape)
    return mask_stack


if __name__ == '__main__':
    torch.manual_seed(1024)
    
    labelpath = r"D:\data\dl_SSD\icn_label_data"
    # labelpath = r"D:\data\dl_SSD\icn_label_data\11"
    datapath = r"D:\data"
    
    # testpath = r"D:\data\2021-03-05\3"
    # trainset_path = r"D:\data\dl_SSD"
    batchsize = 3
    
    dataset = getdata(labelpath, datapath , transform = True)  #true : complex , #false :float
    
    train_size = int(len(dataset)*0.9)
    val_size = len(dataset) - train_size
    trainset, valset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    trainset_loader = DataLoader(trainset , batch_size = batchsize, shuffle = True)
    
    # valset = valdata(labelpath, datapath , transform = True)  #true : complex , #false :float
    valset_loader = DataLoader(valset , batch_size = batchsize , shuffle = True)
    
    # testset = testdata(testpath)   #true : complex , #false :float
    # testset_loader = DataLoader(testset , batch_size = batchsize,num_workers=0, shuffle = False)
    
    model = simple_icnet.ICNet(num_classes=2).to('cuda')
    if os.path.exists(r"simple_icnet_mbg.pt"):
        print('load')
        checkpoint = torch.load(r"simple_icnet_mbg.pt")
        model.load_state_dict(checkpoint['state_dict'])
        
    criterion = ICNetLoss(ignore_index=-1).to('cuda')
    
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001) #0.0001


    train(trainset_loader)
    
    # import time
    
    # t1 = time.time()
    # savepath = testpath+r"\analysis"
    # try:
    #     os.mkdir(savepath)
    # except :
    #     pass

        
    # mask_stack = test(model , savepath)
    


   
    # validate(model)
    # t2 = time.time()
    # print(t2-t1)



