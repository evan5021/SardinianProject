import sys 
import matplotlib.pyplot as plt
from IPython import display

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


def test_against_string_word(net, n_lines=1, prime='import', top_k=None, device='cpu', display=False):
    max_len = 20
    net.to(device)
    net.eval()

    #top_k = 3

    # First off, run through the prime characters
    #top_k_chars = []
    chars1 = []
    chars2 = []
    chars3 = []
    h = None
    for ch in prime:
        #print(net.predict(ch, h, device=device, top_k=top_k))
        char, h, top_k_chars = net.predict(ch, h, device=device, top_k=top_k)
        """
        try:
            char, h, top_k_chars = net.predict(ch, h, device=device, top_k=top_k)
        except ValueError:
            char, h = net.predict(ch, h, device=device, top_k=top_k)
            """

            
    
    chars1.append(top_k_chars[0])
    chars2.append(top_k_chars[1])
    chars3.append(top_k_chars[2])

    h1 = h
    h2 = h
    h3 = h
 #Predicting on first char
    for ii in range(1, max_len):
        res = net.predict(chars1[-1], h1, device=device, top_k=top_k)
        #print('RES:', res)
        char = res[0]
        h1 = res[1]
        #char, h, topch = net.predict(chars[-1], h, device=device, top_k=top_k)


        if char == ' ':
            chars1.append(' ')
            break
        if char == '.' or char == ',':
            break
        else:
            chars1.append(char)
            
 #Predicting on second char            
    for ii in range(1, max_len):
        res = net.predict(chars2[-1], h2, device=device, top_k=top_k)
        #print('RES:', res)
        char = res[0]
        h2 = res[1]
        #char, h, topch = net.predict(chars[-1], h, device=device, top_k=top_k)


        if char == ' ':
            chars2.append(' ')
            break
        if char == '.' or char == ',':
            break
        else:
            chars2.append(char)
            
 #Predicting on third char           
    for ii in range(1, max_len):
        res = net.predict(chars3[-1], h3, device=device, top_k=top_k)
        #print('RES:', res)
        char = res[0]
        h3 = res[1]
        #char, h, topch = net.predict(chars[-1], h, device=device, top_k=top_k)


        if char == ' ':
            chars3.append(' ')
            break
        if char == '.' or char == ',':
            break
        else:
            chars3.append(char)            
            
            
            
    print(Fore.YELLOW + prime + Fore.BLUE + ''.join(chars1) + Style.RESET_ALL, end='\n')
    print(Fore.YELLOW + prime + Fore.BLUE + ''.join(chars2) + Style.RESET_ALL, end='\n')
    print(Fore.YELLOW + prime + Fore.BLUE + ''.join(chars3) + Style.RESET_ALL, end='\n')
    return prime + ''.join(chars1), prime + ''.join(chars2), prime + ''.join(chars3)
          

def clean_string(x):
    return x

cp = sys.argv[1]

net, _ = load_checkpoint(cp)

saved_clicks = 0


files = open("test.txt", "r")

for line in files:
    i = 0
    clean_prefix = clean_string(line)
    while i < len(clean_prefix):
        temp_prefix=clean_prefix[0:i+1]
        print(temp_prefix)
        temp_output = test_against_string_word(net, 1, prime=temp_prefix, top_k=3)
        if temp_output[0] == clean_prefix[0:len(temp_output[0])]:
            print(Fore.GREEN + temp_output[0] + Style.RESET_ALL)
            saved_clicks = saved_clicks + len(temp_output[0]) - (i + 1)
            i = len(temp_output[0])
        elif temp_output[1] == clean_prefix[0:len(temp_output[1])]:
            print(Fore.GREEN + temp_output[1] + Style.RESET_ALL)
            saved_clicks = saved_clicks + len(temp_output[1]) - (i + 1)
            i = len(temp_output[1])
        elif temp_output[2] == clean_prefix[0:len(temp_output[2])]:
            print(Fore.GREEN + temp_output[2] + Style.RESET_ALL)
            saved_clicks = saved_clicks + len(temp_output[2]) - (i + 1)
            i = len(temp_output[2])
        else:
            i = i + 1
            
print(str(saved_clicks))