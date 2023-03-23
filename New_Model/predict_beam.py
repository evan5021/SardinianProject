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

import numpy as np

def beam_search(net, prime, beam_size, max_len, device='cpu'):
    net.to(device)
    net.eval()

    # Initialize hidden state
    h = None
    
    # Initialize sequences with empty list and score 0.0
    sequences = [[[], 0.0, h]]

    # First, prime the network with the given string
    for ch in prime:
        chars, p_list, h = net.predict_beam(ch, h, device=device, top_k=1)
        
    #print(chars, p_list, h)
    char = chars[0]
    probs = p_list[0]
        
    all_candidates = []
    # Iterate until max_len or until all sequences end in a space character
    for _ in range(max_len):
        
        # Iterate over each sequence in the beam
        for seq in sequences:
            #print('seq: ', seq)
            # Get the last character of the sequence
            if len(seq[0]) > 0:
                ch = seq[0][-1]
            else:
                print('here')
                ch = prime[-1]
                seq[2] = h
            # Use the network to predict the next character
            #char_probs, h = net.predict_beam(ch, h, device=device, top_k=5)
            if not ch == ' ':
                top_k_chars, p_list, hi = net.predict_beam(ch, seq[2], device=device, top_k=5)
            #print("top_k_chars: ", top_k_chars, " p_list: ", p_list)
            #Create a tuple list
                tuple_list = list(zip(top_k_chars, p_list, [hi] * len(top_k_chars)))
            
            #This is not what I really want here
                for item in tuple_list:
                    if ' ' in item:
                        tuple_list.remove(item)
            
            #print("tuple list: ", tuple_list)
            # Sort the probabilities in descending order and get the top beam_size indices
                tuple_list = sorted(tuple_list, key=lambda x : x[1], reverse=True)
                #print('tuple list: ', tuple_list)
            #print("tuple list: ", tuple_list)
            # Create a new sequence for each of the top indices
                for i in tuple_list:
                    new_seq = [seq[0] + [i[0]], seq[1] + np.log(i[1]), seq[2]]
                    all_candidates.append(new_seq)
            else:
                all_candidates.append(seq)
                
            print('CANDIDATES - ')
            for item in all_candidates:
                print("[" + ''.join([i for i in item[0]]) + " " + str(item[1]) + "]")
        # Sort all the sequences by score
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        print('highest: ' + str(ordered[0][1]) + ' lowest: ' + str(ordered[-1][1]))
        # Keep only the top beam_size sequences
        sequences = ordered[:beam_size]
        # Check if all the top sequences end in a space character
        #print('sequences are - ', sequences)
        if sequences[0][0] == [] or all((seq[0][-1]) == ' ' for seq in sequences):
            break
    # Get the top sequence and convert it to a string
    top_sequence = sequences[0][0]
    #print(sequences)
    #print('here')
    #print('top: ', top_sequence)
    return prime + ''.join([i for i in top_sequence])

          

def clean_string(x):
    return x

cp = sys.argv[1]

net, _ = load_checkpoint(cp)

saved_clicks = 0

files = open("test.txt", "r")

for line in files:
    i = 0
    clean_prefix = clean_string(line)
    print(clean_prefix)
    while i < len(clean_prefix):
        temp_prefix=clean_prefix[0:i+1]
        print(Fore.YELLOW + temp_prefix + Style.RESET_ALL)
        temp_output = beam_search(net, temp_prefix, 3, 20)
        print(Fore.BLUE + temp_output + Style.RESET_ALL)
        if temp_output == clean_prefix[0:len(temp_output)]:
            print(Fore.GREEN + temp_output)
            saved_clicks = saved_clicks + len(temp_output) - (i + 1)
            i = len(temp_output)
        else:
            i = i + 1
            
print(str(saved_clicks))