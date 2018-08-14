from scipy.stats import norm
import numpy as np
import random
import pandas as pd
import glob

def get_sigcut(x,sigma):
    # identify the sigma cut for a given dataset fitted with a Gaussian distribuion
    param=norm.fit(x)
    range_x=np.linspace(min(x),max(x),1000)
    sigcut = (param[1]*sigma)+param[0]
    return sigcut,param,range_x # return the sigma cut, the Gaussian model and the range fitted over

def load_data(stableData,simulatedData):
    stable_data=pd.read_csv(stableData)
    stable_data['label']='stable'
    stable_data['variable']=0
    all_data=stable_data
    files = glob.glob(simulatedData)
    for filename in files:
        dataTmp = pd.read_csv(filename)
        dataTmp['label']=filename.split('m_')[1].split('_trans_data')[0]
        dataTmp['variable']=1
        all_data=all_data.append(dataTmp)
    return all_data
        
def precision_and_recall(tp,fp,fn):
    # calculate the precision and recall values
    if tp==0:
        precision=0.
    else:
        precision=float(tp)/float(tp+fp)
    if tp==0:
        recall=0.
    else:
        recall=float(tp)/float(tp+fn)
    return precision, recall

def shuffle_datasets(data):
    # shuffle the data into a random order
    shuffled=[]
    val_list=range(len(data))
    random.shuffle(val_list)
    for row in range(len(data)):
        shuffled.append(data[val_list[row]])
    shuffled=np.array(shuffled)
    # returning the shuffled dataset
    return shuffled

def create_datasets(data, n, m):
    # split the data after shuffling
    # n and m are the fraction of the data to be the training dataset and the validation dataset respectively
    shuffle_datasets(data)
    train=data[:n,:]
    valid=data[n:m,:]
    test=data[m:,:]
    # return the training, validation and test datasets 
    return train, valid, test

def write_test_data(filename,data):
    data = np.asarray(data)
    np.savetxt(filename, data, delimiter=",")
    return
