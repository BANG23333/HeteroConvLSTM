import torch
import matplotlib.pyplot as plt
import numpy as np

torch.set_printoptions(edgeitems=100)
def accuracy(vector_x, vector_y):

  # torch.Size([283200])
  new_v = vector_x - vector_y
  new_v = torch.abs(new_v)
  new_v = torch.sum(new_v).data.cpu().numpy()

  return new_v/len(vector_x)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def MSE_torch(prediction, true_value):
  prediction = prediction.flatten(0)
  true_value = true_value.flatten(0)

  prediction = torch.round(prediction)

  mse = torch.sum(torch.square(prediction - true_value)/len(prediction))

  return mse

def plot_loss(train_loss_arr, valid_loss_arr):
  fig, ax1 = plt.subplots(figsize=(20, 10))

  ax1.plot(train_loss_arr, 'k', label='training loss')
  ax1.plot(valid_loss_arr, 'g', label='validation loss')
  ax1.legend(loc=1)
  ax2 = ax1.twinx()
  #ax2.plot(train_mape_arr, 'r--', label='train_mape_arr')
  #ax2.plot(v_mape_arr, 'b--', label='v_mape_arr')

  ax2.legend(loc=2)
  plt.show()
  plt.clf()

def plot_loss(train_loss_arr, valid_loss_arr):
  fig, ax1 = plt.subplots(figsize=(20, 10))

  ax1.plot(train_loss_arr, 'k', label='training loss')
  ax1.plot(valid_loss_arr, 'g', label='validation loss')
  ax1.legend(loc=1)
  ax2 = ax1.twinx()
  #ax2.plot(train_mape_arr, 'r--', label='train_mape_arr')
  #ax2.plot(v_mape_arr, 'b--', label='v_mape_arr')

  ax2.legend(loc=2)
  plt.show()
  plt.clf()

def MSE_np(prediction, true_value):
    prediction = prediction.flatten()
    true_value = true_value.flatten()

    #prediction = np.round(prediction)

    mse = np.sum(np.square(prediction - true_value)/len(prediction))

    return mse