import sys 
import matplotlib.pyplot as plt
from IPython import display # live plotting

from charrnn import *

from datetime import datetime
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

#Color
from colorama import Fore, Back, Style

import re

def load_checkpoint(filename):
    with open(filename, 'rb') as f:
        checkpoint = torch.load(f, map_location='cpu')

    net = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    net.load_state_dict(checkpoint['state_dict'])

    train_history = checkpoint['train_history']

    return net, train_history


"""
def sample_lines(net, n_lines=3, prime='import', top_k=None, device='cpu', max_len=1):
    net.to(device)
    net.eval()

    # First off, run through the prime characters
    chars = []
    h = None
    for ch in prime:
        char, h = net.predict(ch, h, device=device, top_k=top_k)

    chars.append(char)

    # Now pass in the previous character and get a new one
    l = 0
    for ii in range(max_len):
        char, h = net.predict(chars[-1], h, device=device, top_k=top_k)
        chars.append(char)
        if char == '\n':
            l += 1
            if l == n_lines:
                break

    return ''.join(chars)
"""

"""
def test_against_string(net, n_lines=1, prime='import', top_k=None, device='cpu', display=False):
    max_len = len(prime)
    net.to(device)
    net.eval()
    
    # First off, run through the prime characters
    chars = []
    chars.append(prime[0])
    h = None
    for ch in prime:
        char, h = net.predict(ch, h, device=device, top_k=top_k)
        
    # Now compare the predicted character and the actual character and append the correct one
    l = 0
    correct_chars = 0
    for ii in range(1, max_len):
        char, h = net.predict(chars[-1], h, device=device, top_k=top_k)
        if char == prime[ii]:
            if display:
                print(''.join(chars) + Fore.GREEN + char + Style.RESET_ALL)
            chars.append(char)
            correct_chars = correct_chars + 1
        else:
            if display:
                print(''.join(chars) + Fore.RED + char + ' (' + prime[ii] +')' + Style.RESET_ALL)
            chars.append(prime[ii])
        if char == '\n':
            l += 1
            if l == n_lines:
                break

    print(str(correct_chars/max_len) + '% correct\n')
    return ''.join(chars)
"""

def test_against_string_word(net, n_lines=1, prime='import', top_k=None, device='cpu', display=False):
    max_len = 20
    net.to(device)
    net.eval()

    #top_k = 3

    # First off, run through the prime characters
    chars = []
    h = None
    for ch in prime:
        #print(net.predict(ch, h, device=device, top_k=top_k))
        try:
            char, h, top_k_chars = net.predict(ch, h, device=device, top_k=top_k)
        except ValueError:
            char, h = net.predict(ch, h, device=device, top_k=top_k)

    chars.append(char)

    for ii in range(1, max_len):
        res = net.predict(chars[-1], h, device=device, top_k=top_k)
        print('RES:', res)
        char = res[0]
        h = res[1]
        #char, h, topch = net.predict(chars[-1], h, device=device, top_k=top_k)


        if char == ' ':
            chars.append(' ')
            break
        else:
            chars.append(char)
            
            
    print(Fore.YELLOW + prime + Fore.BLUE + ''.join(chars) + Style.RESET_ALL, end='\t')
    return prime + ''.join(chars)
          

def clean_string(x):
    return x

cp = sys.argv[1]

net, _ = load_checkpoint(cp)
     

#prefix= 'normas ortogràficas' # prefix

#clean_prefix = clean_string(prefix)
#print(sample_lines(net, 5, prime=clean_prefix, top_k=3))

#prefix = "s'influentzia semper prus manna de firentzedurante su perìodu de sa famiglia medici, s'umanismu e su rinascimentu ant fatu de su daletu suo o prus chi no àteru de una versione rafinada de custu, unu istandard in is artes."
               
    
#clean_prefix = clean_string(prefix)
#print(test_against_string(net, 1, prime=clean_prefix, top_k=3, display=True))

#from torch.utils.data import Dataset, DataLoader
#
#def conv(predicate_encoder, data, data_train_loader, train_loader, train_loader):
#    plt.model()
#    for item in range(0, n_layers):

#clean_prefix = clean_string(prefix)

#print(clean_prefix)

saved_clicks = 0

"""
i = 0

while i < len(clean_prefix):
    temp_prefix=clean_prefix[0:i+1]
#    print(temp_prefix)
    temp_output = test_against_string_word(net, 1, prime=temp_prefix, top_k=3)
    if temp_output == clean_prefix[0:len(temp_output)]:
        #print("Match!")
        saved_clicks = saved_clicks + len(temp_output) - (i + 1)
        i = len(temp_output)
    else:
        i = i + 1

print("We saved " + str(saved_clicks) + " clicks!")
"""

files = open("test.txt", "r")

for line in files:
    i = 0
    clean_prefix = clean_string(line)
    while i < len(clean_prefix):
        temp_prefix=clean_prefix[0:i+1]
        print(temp_prefix)
        temp_output = test_against_string_word(net, 1, prime=temp_prefix, top_k=3)
        if temp_output == clean_prefix[0:len(temp_output)]:
            print(Fore.GREEN + temp_output)
            saved_clicks = saved_clicks + len(temp_output) - (i + 1)
            i = len(temp_output)
        else:
            i = i + 1
            
print(str(saved_clicks))