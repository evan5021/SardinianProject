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

def test_against_string_word(net, n_lines=1, prime='import', top_k=None, device='cpu', display=False):
    max_len = 20
    net.to(device)
    net.eval()

    # First off, run through the prime characters
    chars = []
    h = None
    for ch in prime:
        char, h = net.predict(ch, h, device=device, top_k=top_k)

    chars.append(char)

    for ii in range(1, max_len):
        char, h = net.predict(chars[-1], h, device=device, top_k=top_k)

        if char == ' ':
            chars.append(' ')
            break
        else:
            chars.append(char)
            
            
    print(Fore.YELLOW + prime + Fore.GREEN + ''.join(chars) + Style.RESET_ALL)
    return prime + ''.join(chars)
          

def clean_string(x):
    return x

cp = sys.argv[1]

net, _ = load_checkpoint(cp)
     

#prefix= 'normas ortogràficas' # prefix

#clean_prefix = clean_string(prefix)
#print(sample_lines(net, 5, prime=clean_prefix, top_k=3))

#prefix = "s'influentzia semper prus manna de firentzedurante su perìodu de sa famiglia medici, s'umanismu e su rinascimentu ant fatu de su daletu suo o prus chi no àteru de una versione rafinada de custu, unu istandard in is artes."
               
    
prefix = "Fillus de Sardigna Bai, amiga, bai, cun is alas di amori, e porta unu saludu a is fillus de Sardigna sparzinàus in su mundu po unu arrogu ‘e pani, negau in domu insoru. Bai, in logus trotus de disisperu e amestura fueddus de cuncordia po is corus in turmentu. Bai, a su fossu ‘e Marcinelle, a is sartus de nebida de is pranuras lombardas, a is fabricas de dolori. Bai, amiga, bai, in terra furistera, e semina arregordusdi argiolas arriendu, cantus di amori, po is corus emigraus. Eo declaro che su titulare de sos deretos d'autore apitzos de custu file est de acordu a la donare in lizentzia d'impreu me in sas tèrmines de su $1. Sa domanda tua non l'amus pòdida protzessare. Custu forsis dipendet de su fatu chi as chircadu una paraula de prus pagu de tres lìteras. O puru forsis as iscritu male sa domanda, pro esempru pische and and asulu. Pro praghere, torra a provare. Totu sas pàginas de Wikipedia sunt disponìbiles cun litzèntzia. Casteddu est una de is tzitades prus antigas de Sardigna, e paret chi dd'aiant fundada sos Fenìtzio sa inghiriu de su sèculu VIII a.C., in s'oru de s'istáinu de Santa Igia. Su numene Limba feniza fiat Caral, numene plurale ca giai dae tempus meda fiat formada dae unas cantas biddas. Is romanos, lompidos in su 238 a.C., dd'aiant fata sea de provìntzia e amanniada, faghende·dda crèschere a inghiriu de sa Pratza de su Cramu fintzas a Santu Sadurru: a foras de sa parte bìvida fiant sos campusantos de ''Tuvixeddu'' (iare cartaginesu) e ''Bonàiri''. In su sèculu II ap.C. dd'ant fata ''colònia''. In edade cristiana at tentu unu pìscamu. In edade romana puru su numene fiat plurale: ''Carales''. Sos bividores depiant èssere unos 30.000. Mancari a serru, Casteddu (tando ''Carales'') at sighidu a bìvere a suta de vàndalos e gregos bizantinos, sende sèmper sa sea de su guvernadore provintziale. In su 711 perou paret chi is arabos dd'apant iscontzada e abruscada, e tando sos bividores ant lassadu sa tzitade romana pro si nche istugiare in una ala prus segura, a s'oru de s'istàinu. A pustis no s'at a intendere prus de ''Carales'', ma de ''Santa Ìgia'' (Santa Gilla, Santa Illa, it. ''Santa Cecilia''). In edade giuigale, Santa Igia fiat sa capitale de su [[Judicadu de Càlaris|Giuigadu (Regnu) de Càlari]]: su numene de sa tzitade antiga si fiat sarbadu in su de su logu. Santa Igia teniat unos 15.000 bividores, una tzinta de muros e una crèsia cadirale: sos rastos si podiant bìdere galu in su '800 in ue oe est sa Tzentrale Elètrica. In su 1216, sa Giuighessa Beneita de Lacon aiat permìtidu a unos cantos pisanos de fraigare unu casteddu in ue oe est su bighinadu de Casteddu de susu. Idea mala, ca in sas gherras intre de Pisa e Gènua su giuigadu si diat èssere alliadu a sa segunda: in su 1258 sos pisanos e sos àteros giuigados sardos alliados issoro iscontzant Santa Igia e nche isparghent su sale. Sos bividores lassant sas ruinas, unos cantos andende a Domunoas e Bidda de Cresia, àteros a fundare Biddanoa, s'apendìtziu de Casteddu chi totu connoschimus. Su casteddu si naraiat Castellu de Castro de Callari, de custu est bènnidu su numene sardu de sa tzitade: Casteddu. Is sardos no podiant bìvere in su Casteddu, ma sceti in sos apendìtzios (''Stampaxi'', ''Biddanoa'', ''Lapola''), e si carchi sardu fiat agatadu a intru de is muros a ghennas serradas (est a nàrrere a su note), fiat acumpangiadu a foras, ghetadu dae sa Turre de Santu Francau (o de carchi àtera). In su 1324 is catalanos ant cunchistadu sa Sardigna pisana, fundende una tzitade in su monte de ''Bonàiri'' (est a narrere àire bona), ma passende giae in su 1325 in Casteddu, boghende is pisanos e intrende·bi sceti bividores catalanos. In sas gherras intre de Aragone e Arbaree, Casteddu dd'ant assitiada sos arbaresos prus bortas, ma no dd'ant mai pigada. Dae su 1410 Casteddu, chi is catalanos nàrant Càller est sa capitale de unu regnu unidu, su ''Regnum Sardiniae et Corsicae'' (Regnu de Sardigna e Corsica). Su re est su de Aragone, apresentadu de unu bisurre o de unu guvernadore. In prus, inoghe s'acorrat su Parlamentu (sos Istamentos). Tando bi depìant esser unos 10.000 abitantes. Fintzas a sa gherra tzivile de su 1700-1714, sa Sardinna est abarrada a suta de s'Aragone a in antis e de sa Ispagna a pustis, cun pagu noas pero de importu: s'universidade (1626), sa Cadirale (pisana, ma torrada a fare in stile barrocu), sa Festa de Sant'Efis, etz. Pigada dae is austrìacos in su 1708 e torrada a cunchistare dae sos ispagnolos in su 1717, est assinnada cun totu sa Sardigna a su Duca de Savoia in su 1720. Su primu guvernadore est istadu su barone de Saint-Rémy (su de su Bastione). Semper abarrada cun is Savoias, oe est su cabulogu de sa Regione Autònoma de sa Sardigna de sa Repùblica Italiana."
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
            print("Match!")
            saved_clicks = saved_clicks + len(temp_output) - (i + 1)
            i = len(temp_output)
        else:
            i = i + 1
            
print(str(saved_clicks))