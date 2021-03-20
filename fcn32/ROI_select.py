# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 09:49:17 2021

@author: YX
"""
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import time
from glob import glob
import os
import argparse
import pickle

#%%
def rand_crop(img, num):
    crops = []  
    size = img.shape[0]
    d_num = size // 512
    pos = np.arange(d_num**2)
    np.random.shuffle(pos)
    pos = pos[:num]
    for i in pos:
        x = i % d_num            
        y = i // d_num
        crop = img[x*1024:x*1024+1024 , y*1024:y*1024+1024]
        crops.append(crop)
        
    return crops , pos

def rand_slice(img):
    size = img.shape[0]
    d_num = size -512
    pos = np.random.randint(d_num)
    crop = img[: , pos:pos+512]
        
    return crop , pos

    
#%%

def mouse_func(event,x,y,flags,param):
    global draw

    if event == cv2.EVENT_RBUTTONDOWN:
        draw = True
        
if os.path.exists(r"D:\data\dl_SSD\roi_allaa.pkl" ):
    with open(r"D:\data\dl_SSD\roi_all.pkl" , "rb") as f:
        ROI = pickle.load(f)
        f.close()
else:
    ROI = {}


path = r"D:\data\2021-02-03\12"
save_p = r"D:\data\dl_SSD"

test_p = r"D:\data\2021-02-03\12\test"
for count , frame in enumerate(sorted(glob(test_p+"\*.bmp"),key=os.path.getmtime)):
    
    if count >= 0 and count % 3 == 0:
        name = frame.replace("D:\\data\\" , "" ).replace("\\","_")
        img = cv2.imread(frame , 0)
        img = cv2.blur(img , (3,3))
        
        # img, pos = rand_slice(img)
        img = img[: , 200:712]
        
        cv2.imwrite(r"D:\data\2021-02-03\12\test\c.bmp" , img)
        # plt.imshow(img)
        # plt.show()
        # np.save(save_p + "/" + name , img)
        
        # img = cv2.resize(img, (512,512), cv2.INTER_AREA)
        ## random crop to 512*512
        
        boxes = []
    
        cv2.namedWindow('Frame',2)
        cv2.resizeWindow('Frame', 100,600) 
        
        # if count > 0:
        #     for box in prev_box:
        #         (x, y, w, h) = [int(v) for v in box]
        #         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
        # cv2.imshow('Frame', img)
        
        stay = True
        b_img = img.copy()
        while stay:
            draw = False
            key = cv2.waitKey(100) & 0xff
            cv2.setMouseCallback('Frame',mouse_func)
            if key == ord('s') or draw == True:
                # 第十一步：选择一个区域，按s键，并将tracker追踪器，frame和box传入到trackers中
                box = cv2.selectROI('Frame', img, fromCenter=False,showCrosshair=True)
     
                boxes.append(box)
    
                
            elif key == ord('r'):
                if len(boxes) >= 1:
                    boxes.pop(-1)
                    b_img = img.copy()
                else:
                    print("nothing left")
                
            elif key == ord('n'):
                np.save(save_p + "/" + name , img)
                ROI.update({name : (boxes,pos)})
                break
            
            elif key == ord('p'):
                break
            
            elif key == ord('q'):
                with open(save_p +"/roi.pkl", 'wb') as f:
                    pickle.dump(ROI, f)
                    f.close()
                print("save at -- "+ str(count))
            
            elif key == ord("o"): 
                mask = np.zeros_like(img)
                for x,y,w,h in boxes:
                    mask[ y:y+h ,x:x+w ] = 1
                    plt.imshow(mask)
                    plt.show()
            
            for box in boxes:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(b_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.imshow('Frame', b_img)
            
        prev_box = boxes
        
        
        if count % 100 == 0:
            with open(save_p +"/roi.pkl", 'wb') as f:
                pickle.dump(ROI, f)
                f.close()

with open(save_p +"/roi.pkl", 'wb') as f:
    pickle.dump(ROI, f)
    f.close()

cv2.destroyAllWindows()

#%%
# path = r"D:\data\2021-02-03\15"

# with open(save_p +"/roi.pkl", 'rb') as f:
#     aa = pickle.load(f)