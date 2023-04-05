#%matplotlib inline
import matplotlib.pyplot as plt
from IPython import display # live plotting

from datetime import datetime
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence

from charrnn import *

# use GPU if available, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# load data
chars, all_vars, data, _ = load_data('data')


import random
SEED = 500

random.seed(SEED)
random.shuffle(data)

data = data[:(int(0.8 * len(data)))]


n_hidden=256 #256
n_layers=3 #3

# create RNN
net = CharRNN(chars, all_vars, n_hidden=n_hidden, n_layers=n_layers)

# train
#plt.figure(figsize=(12, 4))
name1 = 'combined_model_'+str(n_hidden)+'_'+str(n_layers)
train(net, data, epochs=100, n_seqs=4, n_steps=1, lr=0.0001, device=device, val_frac=0.5,
      name = name1, plot=False, early_stop=False)
