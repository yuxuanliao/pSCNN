# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 11:27:02 2021

@author: zmzhang
"""

import nmrglue as ng
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def read_bruker_h(nmr_path, bRaw = False, bMinMaxScale = False):
    nmr_path = os.path.normpath(nmr_path)
    if bRaw:
        dic, fid = ng.fileio.bruker.read(f'{nmr_path}/1')
        zero_fill_size = dic['acqus']['TD']
        fid = ng.bruker.remove_digital_filter(dic, fid)
        fid = ng.proc_base.zf_size(fid, zero_fill_size)
        fid = ng.proc_base.fft(fid)
        fid = ng.proc_autophase.autops(fid, 'acme', **{'disp':False})
        fid = ng.proc_base.rev(fid) 
    else:
        dic, fid = ng.fileio.bruker.read_pdata(f'{nmr_path}/1/pdata/1')
        zero_fill_size = dic['acqus']['TD']
    if bMinMaxScale:
        fid = fid / np.max(fid)
    offset = (float(dic['acqus']['SW']) / 2) - (float(dic['acqus']['O1']) / float(dic['acqus']['BF1']))
    start = float(dic['acqus']['SW']) - offset
    end = -offset
    step = float(dic['acqus']['SW']) / zero_fill_size
    ppms = np.arange(start, end, -step)[:zero_fill_size]
    return {'name':nmr_path.split(os.sep)[-1], 'ppm': ppms, 'fid':fid, 'bRaw': bRaw}


def read_bruker_hs(data_folder, bRaw, bMinMaxScale, bDict):
    if bDict:
        spectra = {}
    else:
        spectra = []
    for name in tqdm(os.listdir(data_folder), desc="Read Bruker H-NMR files"):
        nmr_path = os.path.normpath(os.path.join(data_folder, name))
        s = read_bruker_h(nmr_path, bRaw, bMinMaxScale)
        if bDict:
            spectra[s['name']] = s
        else:
            spectra.append(s)
    return spectra

    