import matplotlib.pyplot as plt
from IPython import display # live plotting
import glob
import os

import sys

from datetime import datetime
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def load_data(path):    
    data_files = glob.glob(path +'/*.txt')	
    data = []
    unencoded_data = []
    all_lines = {}
    all_vars = []
    all_text = '' #Added a var that holds all the text
    i = 0
    
    idx2var = []
    
    for fn in data_files:
        var = os.path.basename(fn).split('.')[0]
        all_lines[i] = open(fn).readlines()
        idx2var.append((var,i))
        for lines in all_lines[i]:
            #print(lines)
            all_text += lines #Add all text to all_text
        all_vars.append(var)
        i += 1

    #chars = set()
    
    chars = tuple(set(all_text)) #Collects a set of all the chars
    
    list_chars = list(chars)
    list_chars.remove(' ')
    list_chars.insert(0, ' ')
    idx2chars = list_chars
    char2idx = {j: i for i, j in enumerate(idx2chars)} 

    for var in all_lines:
        for line in all_lines[var]:
            new_line = []
            for char in line:
                new_line.append(char2idx[char])
            data.append((new_line, var))
            unencoded_data.append((line, idx2var[var]))

    #data.sort(key=lambda x : len(x))  
    #print('DATA: ', data)
    #We want to return an encoded array for 'data'
    #print('New_data: ', new_data)
    return chars, all_vars, data, unencoded_data

"""
def split_data(data):
    text_data = []
    var_data = []
    for sent in data:
        text_sent_data = []
        var_sent_data = []
        for elem in sent:
            text_sent_data.append(elem[0])
            var_sent_data.append(elem[1])
        text_data.append(text_sent_data)
        var_data.append(var_sent_data)
        
    return text_data, var_data
"""
"""
def sort_sublists(data):
    new_data = sorted(data, key=lambda x: 
"""
"""
def pad_seqs(data):
    max_len = len(data[-1])
    for sent in data:
        while len(sent) < max_len:
            sent.append([0, sent[0][1]])
    return data
"""

def one_hot_encode(arr, n_labels):
    """
    One-hot encoding for character-data
    """
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot


def get_batches(data, n_seqs, n_steps): #maybe add another element that is the variant
    """
    Batch generator that returns mini-batches of size (n_seqs x n_steps)
    """
    
    batch_size = n_seqs * n_steps
     
    import random
    random.shuffle(data)
    
    for line in data:
        arr = np.array(line[0])
        var = line[1]
        n_batches = len(arr) // batch_size

        # always create full batches
        arr = arr[:n_batches * batch_size]
    
        # reshape
        arr = arr.reshape((n_seqs, -1))
    
        for n in range(0, arr.shape[1], n_steps):
            # features (sequence of characters)
            x = arr[:, n:n + n_steps]
        
            # targets (the next character after the sequence)
            y = np.zeros_like(x)
            try:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + n_steps]
            except IndexError:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
            
            var_arr = np.zeros_like(y, dtype=y.dtype)
            var_arr.fill(var)
        
            yield x, [y, var_arr]

class CharRNN(nn.Module):
    def __init__(self, tokens, vars_ls, n_hidden=256, n_layers=2, drop_prob=0.5, num_vars = 4):
        """
        Basic implementation of a multi-layer RNN with LSTM cells and Dropout.
        """        
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        #add something like
        #self.n_features = 2
        self.vars = vars_ls
        self.chars = tokens
        self.int2var = dict(enumerate(self.vars))
        self.var2int = {var: ii for ii, var in self.int2var.items()}
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        self.dropout = nn.Dropout(drop_prob)
        
        #Perhaps replace self.lstm... with
        #self.conv = nn.conv2d(len(self.chars), n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)
        self.fc1 = nn.Linear(n_hidden, len(self.chars))
        self.fc2 = nn.Linear(n_hidden, num_vars)
        #in output layer, (n_hidden, for x in range(num_chars) len(self.chars) + num_vars)
             
    def get_n_hidden(self):
        return self.n_hidden
    
    def get_n_layers(self):
        return self.n_layers

    
    #do we need to pass two things here?
    #maybe have two forwards?
    def forward(self, x, hidden): #self, x1, x2, hidden
        """
        Forward pass through the network
        """
        
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        
        x_char = self.fc1(x)
        x_var = self.fc2(x)
        
        return x_char, x_var, hidden
    
    def predict(self, char, hidden=None, device=torch.device('cpu'), top_k=None):
        """
        Given a character, predict the next character. Returns the predicted character and the hidden state.
        """
        
        with torch.no_grad():
            self.to(device)
            try:
                x = np.array([[self.char2int[char]]])
            except KeyError:
                return '', '', hidden, ['', '', '']

            x = one_hot_encode(x, len(self.chars))

            inputs = torch.from_numpy(x).to(device)
            
            out_char, out_var, hidden = self.forward(inputs, hidden) #inputs1, hidden


            p = F.softmax(out_char, dim=2).data.to('cpu')
            
            p1 = F.softmax(out_var, dim=2).data.to('cpu')

            if top_k is None:
                top_ch = np.arange(len(self.chars))
            else:
                p, top_ch = p.topk(top_k)
                top_ch = top_ch.numpy().squeeze()
                
            p1, top_vars = p1.topk(4) #num vars

            top_vars = top_vars.numpy().squeeze()
            p1 = p1.numpy().squeeze()
            
            if top_k == 1:
                char = int(top_ch)
            else:
                p = p.numpy().squeeze()
                char = np.random.choice(top_ch, p=p / p.sum())

            top_k_chars = []
            for i in top_ch:
                top_k_chars.append(self.int2char[i])
            
            out_vars = []
            for i in top_vars:
                out_vars.append(self.int2var[i])
                
            #out_var = self.int2var[out_var]
            
            
            #top_k_chars = [self.int2char[i] for i in top_ch]
            
            #print(self.int2char[char])
            #print(hidden)
            #print('TOP_K_CHARS:', top_k_chars)

            return self.int2char[char], out_vars, hidden, top_k_chars

"""
    def predict_beam(self, char, hidden=None, device=torch.device('cpu'), top_k=None):
        with torch.no_grad():
            self.to(device)
            
            try:
                x = np.array([[self.char2int[char]]])
            except KeyError:
                #return ['', '', ''], [0, 0, 0], hidden
                return [''] * top_k, [0] * top_k, hidden

            x = one_hot_encode(x, len(self.chars))
            inputs = torch.from_numpy(x).to(device)

            out, hidden = self.forward(inputs, hidden)
            
            p = F.softmax(out, dim=2).data.to('cpu')

            if top_k is None:
                top_ch = np.arange(len(self.chars))
            else:
                p, top_ch = p.topk(top_k)
                top_ch = top_ch.numpy().squeeze()
                #print("p: ", p,"top_ch: ", top_ch)
                p_list_temp = p.tolist()
                #print(p_list_temp)
                p_list = []
                for r in range(top_k):
                    p_list.append(p_list_temp[0][0][r])

            if top_k == 1:
                char = int(top_ch)
            else:
                p = p.numpy().squeeze()
                char = np.random.choice(top_ch, p=p / p.sum())

            top_k_chars = []
            #print("top_ch: ", top_ch)
            for i in top_ch.ravel().tolist():
                top_k_chars.append(self.int2char[i])
                
            return top_k_chars, p_list, hidden
"""

def save_checkpoint(net, opt, filename, train_history={}):
    """
    Save trained model to file.
    """
    checkpoint = {'n_hidden': net.n_hidden,
                  'n_layers': net.n_layers,
                  'state_dict': net.state_dict(),
                  'optimizer': opt.state_dict(),
                  'tokens': net.chars,
                  'train_history': train_history,
                  'var_ls': net.vars
                  }

    with open(filename, 'wb') as f:
        torch.save(checkpoint, f)


def load_checkpoint(filename):
    """
    Load trained model from file.
    """
    with open(filename, 'rb') as f:
        checkpoint = torch.load(f, map_location='cpu')

    net = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    net.load_state_dict(checkpoint['state_dict'])

    return net, checkpoint

plt.ion() # Allow live updates of plots

def train(net, data, epochs=10, n_seqs=10, n_steps=50, lr=0.001, clip=5, val_frac=0.1, device=torch.device('cpu'),
          name='checkpoint', early_stop=True, plot=True):
    """
    Training loop.
    """
    net.train() # switch into training mode
    opt = torch.optim.Adam(net.parameters(), lr=lr) # initialize optimizer
    
    criterion_char = nn.CrossEntropyLoss()
    criterion_var = nn.CrossEntropyLoss()
    # initialize loss function

    
    
    #Currently 'data' is a dictionary of vars:[text]
    
    # create training and validation data
    
    net.to(device) # move neural net to GPU/CPU memory
    
    min_val_loss = 10.**10 # initialize minimal validation loss
    train_history = {'epoch': [], 'step': [], 'loss': [], 'val_loss': []}

    n_chars = len(net.chars) # get size of vocabulary
    n_vars = len(net.vars)
    
    
    
    val_idx = int(len(data) * (1 - val_frac))
    
    data, val_data = data[:val_idx], data[val_idx:]
    
    # main loop over training epochs
    for e in range(epochs):
        
        hidden = None # reste hidden state after each epoch
    
        # loop over batches
        for x, y in get_batches(data, n_seqs, n_steps):
            
                # encode data and create torch-tensors
            x = one_hot_encode(x, n_chars)
                
                #print('y[0] ', type(y[0]))
                #print('y[1] ', y[1])
                
            inputs, targets_char, targets_var = torch.from_numpy(x).to(device), torch.tensor(y[0], dtype = torch.long).to(device), torch.tensor(y[1], dtype = torch.long).to(device)
                
                #print('next ', targets_var)
                
                # reset gradient information
            net.zero_grad()
            
                # generate network output
            output_char, output_var, hidden = net.forward(inputs, hidden)
            
            
                # compute loss
            loss_char = criterion_char(output_char.view(n_seqs * n_steps, n_chars), targets_char.view(n_seqs * n_steps))
                
                #print('output_var ', output_var.view(n_seqs * n_steps, n_vars).squeeze())
                #print('targets_var ', targets_var)
                
            loss_var = criterion_var(output_var.view(n_seqs * n_steps, n_vars), targets_var.view(n_seqs * n_steps))
            
            loss = loss_char + loss_var
            
                # compute gradients
            loss.backward()

                # gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            
                # optmize
            opt.step()

                # prevent backpropagating through the entire training history
                # by detaching hidden state and cell state
            hidden = (hidden[0].detach(), hidden[1].detach())
            
            # validation step is done without tracking gradients
        with torch.no_grad():
            val_h = None
            val_losses = []
            
            for x, y in get_batches(val_data, n_seqs, n_steps):
                x = one_hot_encode(x, n_chars)
                inputs, targets_char, targets_var = torch.from_numpy(x).to(device), torch.tensor(y[0], dtype = torch.long).to(device), torch.tensor(y[1], dtype = torch.long).to(device)

                output_char, output_var, val_h = net.forward(inputs, val_h)
                
                val_loss_char = criterion_char(output_char.view(n_seqs * n_steps, n_chars), targets_char.view(n_seqs * n_steps))
                val_loss_var = criterion_var(output_var.view(n_seqs * n_steps, n_vars), targets_var.view(n_seqs * n_steps))
                    
                val_loss = val_loss_char.item() + val_loss_var.item()
                    
                val_losses.append(val_loss)
            
                # compute mean validation loss over batches
            mean_val_loss = np.mean(val_losses)
            
            # track progress
            train_history['epoch'].append(e+1)
            train_history['loss'].append(loss.item())
            train_history['val_loss'].append(mean_val_loss)
        
        if plot:
            # create live plot of training loss and validation loss
            plt.clf()
            plt.plot(train_history['loss'], lw=2, c='C0')
            plt.plot(train_history['val_loss'], lw=2, c='C1')
            plt.xlabel('epoch')
            plt.title("{}   Epoch: {:.0f}/{:.0f}   Loss: {:.4f}   Val Loss: {:.4f}".format(
                datetime.now().strftime('%H:%M:%S'),
                e+1, epochs,
                loss.item(),
                mean_val_loss), color='k')
            display.clear_output(wait=True)
            display.display(plt.gcf())
        else:
            # print training progress
            print("{}   Epoch: {:.0f}/{:.0f}   Loss: {:.4f}   Val Loss: {:.4f}".format(
                datetime.now().strftime('%H:%M:%S'),
                e+1, epochs,
                loss.item(),
                mean_val_loss))
        
        # save model checkpoint if validation loss has decreased
        if mean_val_loss < min_val_loss:
            save_checkpoint(net, opt, name+'.net', train_history=train_history)
            min_val_loss = mean_val_loss
        
        # if validation loss has not decreased for the last 10 epochs, stop training
        if early_stop:
            if e - np.argmin(train_history['val_loss']) > 10:
                display.clear_output()
                print('Validation loss does not decrease further, stopping training.')
                break
    plt.plot(train_history['epoch'], train_history['loss'], label='loss')
    plt.plot(train_history['epoch'], train_history['val_loss'], label='val_loss')
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig('loss_graph_hidden' + str(net.get_n_hidden()) + '_layers' + str(net.get_n_layers()) + '.png')