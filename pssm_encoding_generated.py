# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 21:37:38 2021

@author: TMU
"""
import os
from sklearn.model_selection import StratifiedKFold
import pandas as pd
maxseq = 4981

def generate_dataset(in_dir, out_dir, label):
    for filename in os.listdir(in_dir):
        pssm = open(os.path.join(in_dir, filename)).readlines()[3:-6]
        arr = []
        j = 0
        for line in pssm:
            j += 1
            arr.extend([float(k) for k in line.split()[2:22]])
        for c in range(j, maxseq):
            arr.extend([0]*20)
        out_dir.write(
            f"{label},{','.join(str(x) for x in arr)}\n")
    
pos_dir = 'pssm/snare/cv'
neg_dir = 'pssm/non-snare/cv'        
ftrn = open(os.path.join('dataset', 'pssm.cv.csv'), 'w')
generate_dataset(pos_dir, ftrn, 1)
generate_dataset(neg_dir, ftrn, 0)
ftrn.close()


