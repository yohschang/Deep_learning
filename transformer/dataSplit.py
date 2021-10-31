# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 22:53:58 2021

@author: roy82
"""

import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

standardization = lambda x : (x - np.mean(x)) / np.std(x)
normalization = lambda x, min_v, max_v : (x - min_v)/ (max_v - min_v)

class dataset:
    def __init__(self):
        
        data = yf.download("AAPL",'2001-07-01','2021-07-01')
        
        closeData = data["Close"]
        high = data["High"]
        low = data["Low"]
        openData = data["Open"]
        Volume = data["Volume"]
        
        min_v = np.min(np.min(data[["Close" , "Open","High" , "Low"]]))
        max_v = np.max(np.max(data[["Close" , "Open","High" , "Low"]]))
        
        self.NorCloseData = normalization(closeData, min_v, max_v)
        self.NorOpenData = normalization(openData, min_v, max_v)
        self.NorHigh = normalization(high, min_v, max_v)
        self.NorLow = normalization(low, min_v, max_v)
        self.NorVolume = standardization(Volume)
        
        self.NorData = pd.concat([self.NorHigh , self.NorLow, self.NorCloseData, self.NorOpenData, self.NorVolume] , axis = 1)

    def split_dataframe(self,df, chunk_size = 1): 
        chunks = list()
        num_chunks = len(df) // chunk_size + 1
        for i in range(num_chunks):
            chunks.append(df[i*chunk_size:(i+1)*chunk_size])
        return np.array(chunks)
    
    def create_dataset(self,data):
        X = []
        Y = []
        for d in data:
            if len(d) < 120 : 
                pass
            else:
                for i in range(120, len(d)-5):
                    X.append(d[i-120:i])
                    Y.append(d[i:i+5]["Close"].tolist()) # use 120 days data to predict 5 days data
                    
        return X, Y

    def generate(self):
        piece = 200
        splitData = self.split_dataframe(self.NorData , chunk_size = piece)

        seq_no = np.arange(len(splitData))
        np.random.shuffle(seq_no)
        
        trainData = splitData[seq_no[:-len(seq_no)//10]]
        valData = splitData[seq_no[-len(seq_no)//10:]]
    
                
        trainX, trainY = self.create_dataset(trainData)
        valX, valY = self.create_dataset(valData)
                
        return np.array(trainX),np.array(trainY),np.array(valX), np.array(valY)


if __name__ == "__main__":
    Dataset = dataset()
    trainX,trainY,valX, valY = Dataset.generate()
    print(len(trainX) , len(valX))
    # plt.plot(valX[0]["Close"])
    # plt.show()




#%%
# plt.plot()
# data["Volume"].plot(kind = "bar")
# locs, labels = plt.xticks()
# N = 100
# plt.xticks(locs[::N], data.index[::N].strftime('%Y-%m-%d'))