import os
import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import pandas as pd
import scipy
from scipy import ndimage
import numpy as np
import cv2 as cv
from PIL import Image
import pickle

filenameToPILImage = lambda x: Image.open(x)
normalize = lambda x : (x - np.min(x)) / np.max(x - np.min(x))

class Dataset1(Dataset):
    def __init__(self, csv_path="Blood_data/train.csv", data_dir="Blood_data/train",size = 32,settype="full"):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path)
        l = len(self.data_df)
        if settype=="full":
            pass
        elif settype=="train":
            self.data_df = self.data_df[:(l*4)//5]
            self.data_df = self.data_df.reset_index() 
        elif settype=="valid":
            self.data_df = self.data_df[(l*4)//5:]
            self.data_df = self.data_df.reset_index() 
        else:
            print("settype wrong")
            qwe
            
        
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize((size,size), interpolation=2),
            transforms.ToTensor()
            ])

    def __getitem__(self, index):
        folder_path = self.data_df.loc[index, "dirname"]
        file_path = self.data_df.loc[index, "ID"]
        image = self.transform(os.path.join(self.data_dir, folder_path, file_path))
        # image = torch.squeeze(image)
        # image = image.expand(3,-1,-1)
        label_df = self.data_df.loc[index,["ich","ivh","sah","sdh","edh"]]
        label = torch.FloatTensor(label_df.values.tolist())
        return image, label

    def __len__(self):
        return len(self.data_df)

class TestDataset(Dataset):
    def __init__(self, data_dir="Blood_data/test",size = 32):
        patients=os.listdir(data_dir)
        self.v=[]
        for f in patients:
            for path in os.listdir(os.path.join(data_dir,f)):
                self.v.append((os.path.join(data_dir,f,path),f,path))
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize((size,size), interpolation=2),
            transforms.ToTensor()
            ])

    def __getitem__(self, index):
        filepath,dirname,ID=self.v[index]
        image = self.transform(filepath)
        return image, dirname,ID

    def __len__(self):
        return len(self.v)


class TestDataset2(Dataset):
    def __init__(self, data_dir="Blood_data/test",size = 32):
        patients=os.listdir(data_dir)
        self.v=[]
        for f in patients:
            arr=[]
            for path in os.listdir(os.path.join(data_dir,f)):
                arr.append((os.path.join(data_dir,f,path),f,path))
            arr.sort(key=lambda x: int(x[2].split('.')[0].split('_')[-1]))
            self.v.append(arr)
            
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize((size,size), interpolation=2),
            transforms.ToTensor()
            ])

    def __getitem__(self, index):
        paths = [filepath for filepath,_,_ in self.v[index]]
        dirnames = [dirname for _,dirname,_ in self.v[index]]
        IDs = [ID for _,_,ID in self.v[index]]
        images = [self.transform(path) for path in paths]
        images =  torch.cat(images)
        return images, dirnames,IDs

    def __len__(self):
        return len(self.v)

class Dataset2(Dataset):
    def __init__(self, csv_path="Blood_data/train.csv", data_dir="Blood_data/train",size = 32):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path)
        self.patientlist=list(set(self.data_df["dirname"]))
        self.patientlist.sort()
        # print(self.data_df)
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize((size,size), interpolation=2),
            transforms.ToTensor()
            ])

    def __getitem__(self, index):
        # get dataframe for specific patient 
        patient=self.patientlist[index]
        patient_df = self.data_df.loc[self.data_df["dirname"]==patient]
        # get images 
        folder_path = patient
        file_path_list = patient_df["ID"].tolist()
        image_list = [self.transform(os.path.join(self.data_dir, folder_path, file_path)) for file_path in file_path_list]
        images = torch.cat(image_list)
        #get labels
        label_df = patient_df[["ich","ivh","sah","sdh","edh"]]
        label = torch.FloatTensor(label_df.values)
        
        return images, label

    def __len__(self):
        return len(self.patientlist)
    
def plot():
    dataset = Dataset1(size = 512)
    trainset_loader = DataLoader(dataset , batch_size = 100, shuffle = True)

#%%
class Dataset_aug_single(Dataset):
    def __init__(self, csv_path="Blood_data/train.csv", data_dir="Blood_data/train",size = 32,settype="full",Normalize = False):
        self.size = size
        self.normalize = Normalize
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path)
        self.mask_dict = {}
        l = len(self.data_df)
        
        self.patientlist=list(set(self.data_df["dirname"]))
        self.patientlist.sort()
        P_L = len(self.patientlist)
        
        if settype=="full":
            pass
        elif settype=="train":
            self.data_df = self.data_df[self.data_df["dirname"].isin(self.patientlist[:((P_L*9)//10)])]
            self.data_df = self.data_df.reset_index() 
            
        elif settype=="valid":
            self.data_df = self.data_df[self.data_df["dirname"].isin(self.patientlist[((P_L*9)//10):])]
            self.data_df = self.data_df.reset_index() 
            
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize((size,size), interpolation=2),
            ])
        self.clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        

    def getmask(self , ID):
        patient_df = self.data_df.loc[self.data_df["dirname"]==ID]
        file_path_list = patient_df["ID"].tolist()
        image_list = [self.transform(os.path.join(self.data_dir, ID, file_path)) for file_path in file_path_list]
        images = np.concatenate(image_list).reshape(-1,self.size,self.size)
        img_sum = np.sum(images.copy() , axis = 0)
        img = img_sum.copy()

        img[img>0] = 1
        mask = ndimage.morphology.binary_opening(img,structure=np.ones((9,9)))

        return mask
           
    def __getitem__(self, index):
        folder_path = self.data_df.loc[index, "dirname"]
        file_path = self.data_df.loc[index, "ID"]
        # try : 
        #     if file_path in self.mask_dict:
        #         mask = self.mask_dict[folder_path]
        #     else :
        #         mask = self.getmask(folder_path)
        #         self.mask_dict.update({file_path : mask})
        #         # mask_file = open("mask.pkl" , "wb")
        #         # pickle.dump(self.mask_dict , mask_file)
        #         # mask_file.close()
        # except :
        #     mask = self.getmask(folder_path)
        #     self.mask_dict = {}
        #     self.mask_dict.update({file_path : mask})
        #     # mask_file = open("mask.pkl" , "wb")
        #     # self.mask_dict = pickle.load(mask_file)
        
        image = self.transform(os.path.join(self.data_dir, folder_path, file_path))
        # image= image*mask
        # if self.normalize:
        #     image = normalize(image)
        image = np.array(image) 
        image = self.clahe.apply(image)
        image = transforms.ToPILImage()(image)
        image = transforms.RandomVerticalFlip(p=0.5)(image)
        image = transforms.RandomHorizontalFlip(p=0.5)(image)
        image = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0)(image)
        image = transforms.RandomAffine(degrees=180,scale=(0.8,1.4))(image)
        image = transforms.ToTensor()(image).float()
        # image = image + torch.randn(image.size()) * 1 + image.mean()
        
        label_df = self.data_df.loc[index,["ich","ivh","sah","sdh","edh"]]
        
        label = torch.FloatTensor(label_df.values.tolist())
        
        return image, label

    def __len__(self):
        return len(self.data_df)
    


#%%
class Dataset_aug_patient(Dataset):
    def __init__(self, csv_path=r"Blood_data/train.csv", data_dir=r"Blood_data/train",settype = "train",size = 32,Normalize = False):
        self.size = size
        self.normalize = Normalize
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path)
        self.settype=settype
        self.patientlist=list(set(self.data_df["dirname"]))
        self.patientlist.sort()
        P_L = len(self.patientlist)
        if settype=="full":
            pass
        elif settype=="train":
            self.patientlist = self.patientlist[:((P_L*9)//10)]

        elif settype=="valid":
            self.patientlist = self.patientlist[((P_L*9)//10):]

        # print(self.data_df)
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize((size,size), interpolation=2),
             transforms.ToTensor()
            ])
        
    def mask(self , img_stack):
        img_sum = np.sum(img_stack.copy() , axis = 2)
        img = img_sum.copy()
        img[img>0] = 1
        mask = ndimage.morphology.binary_opening(img,structure=np.ones((9,9)))

        masks = np.repeat(mask[:, :, np.newaxis],img_stack.shape[2], axis=2)
        img_stack = img_stack*masks 
        # if self.normalize:
        #     img_stack = self.stack_normalize(img_stack)
        
        img_stack = transforms.ToTensor()(img_stack).float()
        # img_stack = img_stack + torch.randn(img_stack.size()) * 1 + img_stack.mean()
        
        return img_stack
    
    def stack_normalize(self,img_stack):
        stack_min = np.min(np.min(img_stack,axis = 0),axis = 0)
        stack_min = stack_min[np.newaxis , np.newaxis,:]
        stack_max = np.max(np.max(img_stack-stack_min,axis = 0),axis = 0)
        stack_max = stack_max[np.newaxis , np.newaxis,:]
        
        return (img_stack - stack_min / stack_max)
        

    def __getitem__(self, index):
        # get dataframe for specific patient 
        patient=self.patientlist[index]
        patient_df = self.data_df.loc[self.data_df["dirname"]==patient]
        # get images 
        folder_path = patient
        file_path_list = patient_df["ID"].tolist()
        image_list = [self.transform(os.path.join(self.data_dir, folder_path, file_path)) for file_path in file_path_list]
        image = torch.cat(image_list)
        image = transforms.RandomVerticalFlip(p=0.5)(image)
        image = transforms.RandomHorizontalFlip(p=0.5)(image)
        image = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0)(image)
        image = transforms.RandomAffine(degrees=180,scale=(0.8,1.4))(image)
        
        #get labels
        label_df = patient_df[["ich","ivh","sah","sdh","edh"]]
        label = torch.FloatTensor(label_df.values)
        
        return image, label

    def __len__(self):
        return len(self.patientlist)
    
#%%
class TestDataset_aug(Dataset):
    def __init__(self, data_dir="Blood_data/test",size = 32):
        patients=os.listdir(data_dir)
        self.size = size
        self.v=[]
        for f in patients:
            arr=[]
            for path in os.listdir(os.path.join(data_dir,f)):
                arr.append((os.path.join(data_dir,f,path),f,path))
            arr.sort(key=lambda x: int(x[2].split('.')[0].split('_')[-1]))
            self.v.append(arr)
            
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize((size,size), interpolation=2),
            ])

    def mask(self , img_stack):
        img_sum = np.sum(img_stack.copy() , axis = 2)
        img = img_sum.copy()
        img[img>0] = 1
        mask = ndimage.morphology.binary_opening(img,structure=np.ones((9,9)))
        # import matplotlib.pyplot as plt 
        # plt.imshow(mask , cmap = "jet")
        # plt.show()
    
        masks = np.repeat(mask[:, :, np.newaxis],img_stack.shape[2], axis=2)
        img_stack = img_stack*masks 
        # img_stack = self.stack_normalize(img_stack)
        image = transforms.RandomRotation((0,360))(image)
        image = transforms.RandomVerticalFlip(p=0.5)(image)
        image = transforms.RandomHorizontalFlip(p=0.5)(image)
        
        img_stack = transforms.ToTensor()(img_stack).float()
        
        return img_stack

    def stack_normalize(self,img_stack):
        stack_min = np.min(np.min(img_stack,axis = 0),axis = 0)
        stack_min = stack_min[np.newaxis , np.newaxis,:]
        stack_max = np.max(np.max(img_stack-stack_min,axis = 0),axis = 0)
        stack_max = stack_max[np.newaxis , np.newaxis,:]
        
        return (img_stack - stack_min / stack_max)
    
    def __getitem__(self, index):
        paths = [filepath for filepath,_,_ in self.v[index]]
        dirnames = [dirname for _,dirname,_ in self.v[index]]
        IDs = [ID for _,_,ID in self.v[index]]
        image_list = [self.transform(path) for path in paths]
        images = np.concatenate(image_list).reshape(-1,self.size,self.size).transpose(1,2,0)
        images = self.mask(images)

        return images, dirnames,IDs

    def __len__(self):
        return len(self.v)   

#%% show reading result
# import matplotlib.pyplot as plt 
# csv_path  = r"C:\Users\YH\Desktop\CVDL\final\medical-imaging-wowjenniferlopez\Blood_data/test.csv"
# data_dir= r"C:\Users\YH\Desktop\CVDL\final\medical-imaging-wowjenniferlopez\Blood_data/train"
# # dataset = Dataset_aug(csv_path=csv_path , data_dir=data_dir , size = 224)
# # dataset = Dataset2(csv_path=csv_path , data_dir=data_dir , size = 224)
# dataset = TestDataset_aug( data_dir=data_dir , size = 224)
# loader = DataLoader(
#     dataset, 
#     batch_size=1,
#     shuffle=False
#     )

# dataiter = iter(loader)
# image, dirs , ids = dataiter.next()
# import matplotlib.pyplot as plt 
# img = image[0,5,:,:].numpy()

# plt.imshow(img , cmap = "jet")
# plt.show()
# def imshow(img):
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1,2,0)) , cmap = "jet")
#     plt.show()
# import torchvision
# imshow(torchvision.utils.make_grid(image))
#%%

if __name__ == "__main__":
    # plot()
    ds = Dataset_aug(settype="valid")
    cases=np.zeros(5)

    loader = DataLoader(
        ds, 
        batch_size=1024,
        num_workers=0,
        shuffle=False
        )
    for i,(_,l) in enumerate(loader):
        cases+=torch.sum(l,0).numpy()
        print(i)
    print("cases_num:")
    print(cases)
    print("cases_probability:")
    print(cases/len(ds))
