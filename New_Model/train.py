#%matplotlib inline
import matplotlib.pyplot as plt
from IPython import display # live plotting

from datetime import datetime
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
     

from charrnn import *

# use GPU if available, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# load data
chars, data = load_data('partial_camp.txt')

#print(data)
#print(chars)

n_hidden=256
n_layers=3

# create RNN
net = CharRNN(chars, n_hidden=256, n_layers=3)

# train
#plt.figure(figsize=(12, 4))
name1 = 'camp_model_'+str(n_hidden)+'_'+str(n_layers)
train(net, data, epochs=100, n_seqs=4, n_steps=1, lr=0.0001, device=device, val_frac=0.5,
      name = name1, plot=False, early_stop=False)
