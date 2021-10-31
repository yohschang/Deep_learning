# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 22:50:17 2021

@author: YX
"""
import numpy as np
import matplotlib.pyplot as plt
import os 
from glob import glob
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from dataSplit import dataset
from Tranformer import create_model
import warnings
warnings.filterwarnings('ignore')

#%% generate data
Dataset = dataset()
trainX,trainY,valX, valY = Dataset.generate()

#%% create model

model = create_model(120,5)
model.summary()

callback = tf.keras.callbacks.ModelCheckpoint('Transformer+TimeEmbedding.hdf5', 
                                              monitor='val_loss', 
                                              save_best_only=True, verbose=1)

history = model.fit(trainX,trainY, 
                    batch_size=16, 
                    epochs=35, 
                    callbacks=[callback],
                    validation_data=(valX, valY))  


#%%
###############################################################################
'''Calculate predictions and metrics'''

#Calculate predication for training, validation and test data
train_pred = model.predict(trainX)
val_pred = model.predict(valX)

#Print evaluation metrics for all datasets
train_eval = model.evaluate(trainX, trainY, verbose=0)
val_eval = model.evaluate(valX, valY, verbose=0)
print(' ')
print('Evaluation metrics')
print('Training Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(train_eval[0], train_eval[1], train_eval[2]))
print('Validation Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(val_eval[0], val_eval[1], val_eval[2]))
print('Test Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(test_eval[0], test_eval[1], test_eval[2]))

#%%

# model = tf.keras.models.load_model('Transformer+TimeEmbedding.hdf5',
#                                    custom_objects={'Time2Vector': Time2Vector, 
#                                                    'SingleAttention': SingleAttention,
#                                                    'MultiAttention': MultiAttention,
#                                                    'TransformerEncoder': TransformerEncoder})

train_pred = model.predict(trainX)
#%%


plt.plot(train_pred)
plt.plot(trainY)
plt.show()



