#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import math
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import random 
import sys
from model import *
from utils import *
from dataset import *


# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[3]:


# X data shape [batch, input_len, x_len_of_grid, y_len_of_grid, features]
# Y data shape [batch, output_len (1), x_len_of_grid, y_len_of_grid]

# loading traninig and validating data.
X = np.load(open('E:/Hetero-convlstm/Data/X_Iowa_2016-2017_7_1.npy', 'rb')) # Specify your data path.
Y = np.load(open('E:/Hetero-convlstm/Data/Y_Iowa_2016-2017_7_1.npy', 'rb')) # Specify your data path.

X, Xv, Y, Yv = train_test_split(X, Y, test_size=0.20, random_state=1)

x_len = X.shape[2] # x_len of your gird
y_len = X.shape[3] # y_len of your grid

mask_table = np.load(open('mask_128_64.npy', 'rb')) # if no mask available use code below to intialize a one matrix
mask = mask_table

# intialize a one matrix
"""
mask = np.ones((x_len, y_len))
mask_table = mask
"""
# loading testing data.
Xt = np.load(open('E:/HintNet/Data/X_Iowa_2018_7_1.npy', 'rb')) # Specify your data path.
Yt = np.load(open('E:/HintNet/Data/Y_Iowa_2018_7_1.npy', 'rb')) # Specify your data path.


# In[4]:


final_prediction = np.zeros((Yt.shape))
final_counter = np.zeros((128, 64))

start_x, start_y = 0, 0
for x in range(7):
  for y in range(3):
    
    flag = 0

    print("model: (" + str(x) + "," + str(y) + ")")
    start_x = x*16
    start_y = y*16
    end_x = start_x + 31
    end_y = start_y + 31

    sub_X = X[:,:,start_x:end_x+1, start_y:end_y+1,:]
    sub_Xv = Xv[:,:,start_x:end_x+1, start_y:end_y+1,:]
    sub_Xt = Xt[:,:,start_x:end_x+1, start_y:end_y+1,:]
    
    sub_Y = Y[:,:,start_x:end_x+1, start_y:end_y+1]
    sub_Yv = Yv[:,:,start_x:end_x+1, start_y:end_y+1]
    sub_Yt = Yt[:,:,start_x:end_x+1, start_y:end_y+1]
    
    final_counter[start_x:end_x+1, start_y:end_y+1] += 1

    while flag == 0:

      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      learning_rate = 0.0001
      model = EncoderDecoderConvLSTM(32, 47).to(device)

      criterion = nn.MSELoss()
      #criterion = nn.BCELoss()

      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
      batch_size = 128
      num_epochs = 1000

      train_loss_arr = []
      train_acc_arr = []
      valid_loss_arr =[]
      valid__acc_arr = []
      test_loss_arr =[]
      test__acc_arr = []

      train_dataset = DM_Dataset(sub_X, sub_Y)
      training_generator = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

      validation_dataset = DM_Dataset(sub_Xv, sub_Yv)
      validation_generator = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

      test_dataset = DM_Dataset(sub_Xt, sub_Yt)
      test_generator = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

      training_counter = 0
      same_counter = 0
      min_valid_loss = 10000
      for echo in range(num_epochs):
        model.train()
        avg_train_loss = []
        avg_train_acc = []
        for local_batch, local_labels in training_generator:
          local_batch, local_labels = local_batch.to(device), local_labels.to(device)

          outputs = model(local_batch, 1)

          outputs = torch.flatten(outputs)
          true_y = torch.flatten(local_labels)

          #t_out = (outputs>0.5).float()
          
          #t_acc = criterion(t_out, true_y)
          t_acc = MSE_torch(outputs, true_y)
          train_loss = criterion(outputs, true_y)


          avg_train_acc.append(t_acc)
          avg_train_loss.append(train_loss.cpu().data)
          optimizer.zero_grad()
          train_loss.backward()
          optimizer.step()
          
        train_loss_arr.append(sum(avg_train_loss) / len(avg_train_loss))
        train_acc_arr.append(sum(avg_train_acc) / len(avg_train_acc))


        model.eval()
        avg_valid_loss = []
        avg_valid_acc = []
        with torch.no_grad():
          for local_batch, local_labels in validation_generator:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            Voutputs = model(local_batch, 1)
            Voutputs = torch.flatten(Voutputs)

            V_true_y = torch.flatten(local_labels)
            #V_out = (Voutputs>0.5).float()

            #acc = criterion(V_out, V_true_y)
            acc = MSE_torch(Voutputs, V_true_y)
            v_loss = criterion(Voutputs, V_true_y)
            
            avg_valid_acc.append(acc)
            avg_valid_loss.append(v_loss.cpu().data)


          valid_loss_arr.append(sum(avg_valid_loss) / len(avg_valid_loss))
          valid__acc_arr.append(sum(avg_valid_acc) / len(avg_valid_acc))

        #if echo % 50 == 0:
        # if echo != 0:
          #  plot_loss(train_loss_arr, valid_loss_arr)

        if valid_loss_arr[-1].item() > min_valid_loss:
          training_counter += 1
        elif min_valid_loss > valid_loss_arr[-1].item():
          temp_prediction = np.zeros((Yt.shape))
          flag = 1
          #print("echo: " + str(echo))
          #print("train_loss: "+str(train_loss_arr[-1].item())  + "||" + "v_loss: " + str(valid_loss_arr[-1].item()))
          # save pred
          out_train_y = model(Variable(torch.Tensor(sub_Xt).float()).to(device), 1)
          out_train_y = torch.flatten(out_train_y)

          out_train_y = torch.reshape(out_train_y, (len(sub_Yt), 1, 32, 32))
          out_train_y = out_train_y.detach().cpu().numpy()

          temp_prediction[:,:,start_x:end_x+1, start_y:end_y+1] = out_train_y[:,:,0:32,0:32]
          training_counter = 0

        elif round(valid_loss_arr[-1].item()) == round(min_valid_loss):
          same_counter += 1

        min_valid_loss = min(min_valid_loss, valid_loss_arr[-1].item())

        if same_counter > 10:
          print("reinitialze model")
          flag = 0
          break           

        if training_counter > 10:
          flag = 1
          print("early_stop mol")
          break

      final_prediction[:,:,start_x:end_x+1, start_y:end_y+1] += temp_prediction[:,:,start_x:end_x+1, start_y:end_y+1]
      #print("train_loss: "+str(train_loss_arr[-1].item())  + "||" + "v_loss: " + str(valid_loss_arr[-1].item()))
      plot_loss(train_loss_arr, valid_loss_arr)

temp = final_prediction
temp = final_prediction/final_counter


# In[5]:


temp = np.where(mask_table==1, temp, 0)
Yt = np.where(mask_table==1, Yt, 0)

print("testing finished")
print("MSE: " + str(MSE_np(temp, Yt)))

