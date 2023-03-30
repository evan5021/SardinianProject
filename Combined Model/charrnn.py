#%matplotlib inline
import matplotlib.pyplot as plt
from IPython import display # live plotting
import glob

import sys

from datetime import datetime
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
     


def get_chars(filepath):
    """
    Returns list of all characters present in a file
    """
    with open(filepath, 'r') as f:
        data = f.read()

    chars = list(set(data))
    return chars


def load_data(filepath, chars=None):
    """
    Opens a data file, determines the set of characters present in the file and encodes the characters.
    """
    with open(filepath, 'r') as f:
        data = f.read()

    if chars is None:
        chars = tuple(set(data))
    
    # lookup tables for encoding
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    
    # encoding
    encoded = np.array([char2int[ch] for ch in data])

    return chars, encoded

def findFiles(path): return glob.glob(path)

def readLines(filename):
    lines = open(filename).read().strip().split('\n')
    return lines

def load_data_cl(chars=None):
    """
    Opens a data file, determines the set of characters present in the file and encodes the characters.
    """
    
    category_lines = {}
    all_categories = []
    for filename in findFiles('data/*.txt'):
        category = filename.split('/')[-1].split('.')[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)
    
    """
    data = ''
    for item in category_lines:
        #print('item: ', item)
        for line in category_lines[item]:
            data = data + line + '\n'
    """
    
        #print('data: ', data)
    if chars is None:
        # Code to get unique characters
        unique_chars = set()
        for value in category_lines.values():
            unique_chars.update(set(value))
        chars = tuple(unique_chars)
        #chars = tuple(set(data))
    
    # lookup tables for encoding
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    
    # encoding
    encoded_dict = {}
    for value in category_lines:
        encoded = np.array([char2int[ch] for ch in category_lines[value]])
        encoded_dict[value] = encoded
    
    
    return chars, all_categories, encoded_dict


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


def get_batches(arr, n_seqs, n_steps):
    """
    Batch generator that returns mini-batches of size (n_seqs x n_steps)
    """
    batch_size = n_seqs * n_steps
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
        yield x, y

class CharRNN(nn.Module):
    def __init__(self, tokens, vars_ls, n_hidden=256, n_layers=2, drop_prob=0.5, num_vars = 4):
        """
        Basic implementation of a multi-layer RNN with LSTM cells and Dropout.
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.vars = vars_ls
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        self.dropout = nn.Dropout(drop_prob)
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)
        self.fc1 = nn.Linear(n_hidden, len(self.chars))
        self.fc2 = nn.Linear(n_hidden, num_vars)
        #in output layer, (n_hidden, for x in range(num_chars) len(self.chars) + num_vars)
             
    def get_n_hidden(self):
        return self.n_hidden
    
    def get_n_layers(self):
        return self.n_layers

    def forward(self, x, hidden):
        """
        Forward pass through the network
        """

        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        x_char = self.fc1(x)
        
        x_var = self.fc2(x)

        return x_var, x_char, hidden

    def predict(self, char, hidden=None, device=torch.device('cpu'), top_k=None):
        """
        Given a character, predict the next character. Returns the predicted character and the hidden state.
        """
        with torch.no_grad():
            self.to(device)
            try:
                x = np.array([[self.char2int[char]]])
            except KeyError:
                return '', hidden, ['', '', '']

            x = one_hot_encode(x, len(self.chars))
            inputs = torch.from_numpy(x).to(device)

            out_char, out_var, hidden = self.forward(inputs, hidden)

            p = F.softmax(out, dim=2).data.to('cpu')

            if top_k is None:
                top_ch = np.arange(len(self.chars))
            else:
                p, top_ch = p.topk(top_k)
                top_ch = top_ch.numpy().squeeze()

            if top_k == 1:
                char = int(top_ch)
            else:
                p = p.numpy().squeeze()
                char = np.random.choice(top_ch, p=p / p.sum())

            top_k_chars = []
            for i in top_ch:
                top_k_chars.append(self.int2char[i])
            #top_k_chars = [self.int2char[i] for i in top_ch]
            
            #print(self.int2char[char])
            #print(hidden)
            #print('TOP_K_CHARS:', top_k_chars)

            return self.int2char[char], out_var, hidden, top_k_chars

    def predict_beam(self, char, hidden=None, device=torch.device('cpu'), top_k=None):
        """
        Given a character, predict the next character. Returns the predicted character and the hidden state.
        """
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

def save_checkpoint(net, opt, filename, train_history={}):
    """
    Save trained model to file.
    """
    checkpoint = {'n_hidden': net.n_hidden,
                  'n_layers': net.n_layers,
                  'state_dict': net.state_dict(),
                  'optimizer': opt.state_dict(),
                  'tokens': net.chars,
                  'train_history': train_history}

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
    criterion = nn.CrossEntropyLoss() # initialize loss function

    # create training and validation data
    val_idx = int(len(data) * (1 - val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    
    net.to(device) # move neural net to GPU/CPU memory
    
    min_val_loss = 10.**10 # initialize minimal validation loss
    train_history = {'epoch': [], 'step': [], 'loss': [], 'val_loss': []}

    n_chars = len(net.chars) # get size of vocabulary
    n_vars = len(net.vars)
    
    # main loop over training epochs
    for e in range(epochs):
        hidden = None # reste hidden state after each epoch
        
        # loop over batches
        for x, y in get_batches(data, n_seqs, n_steps):

            # encode data and create torch-tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x).to(device), torch.tensor(y, dtype=torch.long).to(device)
            
            # reset gradient information
            net.zero_grad()
            
            # generate network output
            output_char, output_var, hidden = net.forward(inputs, hidden)
            
            # compute loss
            loss_char = criterion(output_char.view(n_seqs * n_steps, n_chars), targets.view(n_seqs * n_steps))
            
            #replaced n_chars with num_vars
            loss_var = criterion(output_var.view(n_seqs * n_steps, n_vars), targets.view(n_seqs * n_steps))
            
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
                inputs, targets = torch.from_numpy(x).to(device), torch.tensor(y, dtype=torch.long).to(device)
                
                output_char, output_var, val_h = net.forward(inputs, val_h)
                #output, val_h = net.forward(inputs, val_h)
                
                val_loss1 = criterion(output.view(n_seqs * n_steps, n_chars), targets.view(n_seqs * n_steps))
                val_loss2 = criterion(output.view(n_seqs * n_steps, n_vars), targets.view(n_seqs * n_steps))
                
                
                
                val_losses.append(val_loss1.item() + val_loss2.item())
            
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