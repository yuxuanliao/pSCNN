import random
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt

def augment_positive(spectra, ids4p, low=0.2, high=1.0):
    no = random.randint(0, len(ids4p)-1) 
    ratio = random.uniform(low, high)   
    x = ratio * spectra[ids4p[-1]]['fid']  
    for i in range(no):                
        ratio = random.uniform(low, high)
        x = x + ratio * spectra[ids4p[i]]['fid']
    return x


def augment_negative(spectra, ids4n, low=0.2, high=1.0):
    no = random.randint(1, len(ids4n)-1)  
    x = np.zeros_like(spectra[0]['fid'])  
    for i in range(no):                   
        ratio = random.uniform(low, high)
        x = x + ratio * spectra[ids4n[i]]['fid']
    return x


def data_augmentation(spectra, n, max_pc, noise_level = 0.001):
    p  = spectra[0]['ppm'].shape[0]   
    s  = len(spectra)
    Rp = np.zeros((n, p), dtype = np.float32)
    Sp = np.zeros((n, p), dtype = np.float32)
    Rn = np.zeros((n, p), dtype = np.float32)
    Sn = np.zeros((n, p), dtype = np.float32)
    for i in tqdm(range(n), desc="Data augmentation"):
        n1 = np.random.normal(0, 1, p)
        n2 = np.random.normal(0, 1, p)
        n3 = np.random.normal(0, 1, p)
        n4 = np.random.normal(0, 1, p)
        ids4p   = random.sample(range(0, s-1), max_pc)
        Rp[i, ] = spectra[ids4p[-1]]['fid'] + (n1-np.min(n1))*noise_level
        Sp[i, ] = augment_positive(spectra, ids4p) + (n2-np.min(n2))*noise_level
        ids4n   = random.sample(range(0, s-1), max_pc+1)
        Rn[i, ] = spectra[ids4n[-1]]['fid'] + (n3-np.min(n3))*noise_level
        Sn[i, ] = augment_negative(spectra, ids4n) + (n4-np.min(n4))*noise_level
    R = np.vstack((Rp, Rn))
    S = np.vstack((Sp, Sn))
    y  = np.concatenate((np.ones(n, dtype = np.float32), np.zeros(n, dtype = np.float32)), axis=None)
    R, S, y = shuffle(R, S, y)
    return {'R':R, 'S':S, 'y':y}


def plot_augment(aug, ids):
    for i in ids:
        fig = plt.figure(figsize=(12,4)) 
        ax = fig.add_subplot(1,1,1) 
        ax.plot(aug['R'][i,], 'r', label = 'R') 
        ax.plot(aug['S'][i,], 'k', label = 'S') 
        ax.set_title(f"{aug['y'][i]}")
        ax.legend()