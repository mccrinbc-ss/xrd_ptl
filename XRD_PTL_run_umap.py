
from argparse import ArgumentParser
# from tqdm import tqdm

import pytorch_lightning as pl 
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import StochasticWeightAveraging

import numpy as np
# import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset as ParentDataset
from torch.utils.data import DataLoader, random_split, Subset

import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.fft import rfft
from scipy.ndimage import gaussian_filter1d

from tqdm import tqdm

pl.utilities.seed.seed_everything(seed = 1154) #3654
pd.options.mode.chained_assignment = None

import XRD_PTL
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error

import joblib




def umap_preprocessing(create_umaps = False, random_seed = 0, y_min = -5, y_max=0):
    print('blast off! umap preprocessing...')

    paths = ['./test_data/batch0_all_data.feather',
    './test_data/batch1_all_data.feather',
    './test_data/batch2_all_data.feather', 
    './test_data/batch3_all_data.feather'
    ]

    umap_embedder_path = 'Images/UMAP/fourier_InP_removed/RI_all_data/not_scaled/n_neigh50/umap_transformer.gz'
    umap_transformer = joblib.load(umap_embedder_path)

    if create_umaps == False:
        umaps = np.load('umaps_batch0123.npy', allow_pickle=True)
    else:
        umaps = [] #keep the set of umaps in memory so we can call them later if we need them. 


    training_indicies_batch = [] #training indicies for EACH batch seperatly. 
    fts = []
    labels_ = []
    for ii, path in enumerate(paths):
        print(f'Umap Batch {ii}...')
        data, labels, ft, spectras_sc, labels_sc, ft_sc  = XRD_PTL.process_data(path, label_location = None, fourier_flag = True, K = 300, indicies_to_keep = False)

        if create_umaps == True:
            print(f'computing umap {path}')
            transform = umap_transformer.transform(ft)
            umaps.append(transform)

        umap = pd.DataFrame(umaps[ii], columns = ['x','y']) #taking the batch of data into the umap space. 
        ix = (umap.x.between(-5,8) & umap.y.between(y_min,y_max)) 
        training_indicies = umap.loc[ix].index
        training_indicies_batch.append(training_indicies)

        fts.append(ft[training_indicies])
        labels_.append(labels[training_indicies])

    ft = np.vstack(fts) #The size of the LARGEST array is 66515
    labels = np.vstack(labels_)

    print(ft.shape)
    np.random.seed(random_seed)
    idx = np.random.choice(ft.shape[0], 66515, replace=False) 
    ft = ft[idx, :] #100,000 samples over the set of possible fts. Therefore, all sets have the same number of samples. 
    labels = labels[idx, :]

    return data, labels, ft


#Create a set of arguments
dict_args = {
    'model_name':'fourier',
    'data_dir':'/home/brian/projects/XRD_PTL/test_data/batch0_all_data.feather',
    'label_dir':'home/brian/projects/XRD_PTL/test/labels.feather',
    'K':300,
    'batch_size':128,
    'num_workers':16,
    'lr': 0.001,
    'dropout':0 ,
    'momentum': 0.1,
    'recon_weight':1,
    'gpus': 1,
    'max_epochs': 710
}

# print('Recon Weight: ' + str(dict_args.get('recon_weight')/10.0))

model_name = dict_args.get('model_name')
label_dir = dict_args.get('label_dir')
data_dir = dict_args.get('data_dir')
K = dict_args.get('K') #number of fourier comps

if model_name == "new_AE":
    model = XRD_PTL.AutoencoderXRD_newmodel(**dict_args)
    fourier_flag = False
elif model_name == "old_AE":
    model = XRD_PTL.AutoencoderXRD_oldmodel(**dict_args)
    fourier_flag = False
elif model_name == "fourier":
    model = XRD_PTL.AutoencoderXRD_fourier(**dict_args)
    fourier_flag = True
else:
    print('No valid model selected.')

# ys = [[-5,0],[-5,5],[-5,10],[-5,15]] #The set of umap y limits. 
ys = [[-5,5],[-5,15]] #The set of umap y limits. 

for ii in range(0, len(ys)): #1 because we already did the first set. 

    print(f'Running phase space experiment {ii}')
    y_min, y_max = ys[ii]
    data, labels, ft = umap_preprocessing(random_seed = ii + 2, y_min = y_min, y_max = y_max) #This function calls process_data

    np.save(f'ft_labels_for_umap_testing/ft_{ii}_{y_max}.npy', ft)
    np.save(f'ft_labels_for_umap_testing/labels_{ii}_{y_max}.npy', labels)

    model_name = f'{model_name}'
    logger = TensorBoardLogger("lightning_logs", name = 'phase_space_testing_v2', default_hp_metric = False)

    # data, labels, ft, spectras_sc, labels_sc, ft_sc  = XRD_PTL.process_data(data_dir, label_location = label_dir, fourier_flag = fourier_flag, K = K, indicies_to_keep = False)

    datasets = XRD_PTL.XRD_dataset_2(data, labels, ft, fourier_flag, single_sample = False, rep_factor = 1)

    train_frac, val_frac, test_frac = (0.7, 0.30, 0.0)
    total_samples = len(datasets) #use a length in the future. 

    train_num = int(total_samples * train_frac)
    val_num = int(total_samples * val_frac)
    test_num = total_samples - train_num - val_num

    train_set, val_set, test_set = random_split(datasets, [train_num, val_num, test_num]) #this will only work for single batches being fed in

    # Create the dataloaders needed for PyTorch
    BS = dict_args.get('batch_size')
    NW = dict_args.get('num_workers')

    #overwriting for batch testing:
    # train_subset = Subset(datasets, [0,1])
    # val_subset = Subset(datasets, [2,3])

    train_loader = DataLoader(train_set, BS, shuffle = True, num_workers = NW) #substitute train_subset for train_set when appropriate
    val_loader = DataLoader(val_set, BS, shuffle = False, num_workers = NW)
    # test_loader = DataLoader(test_set, BS, shuffle = False, num_workers = NW)

    trainer = pl.Trainer(max_epochs= dict_args.get('max_epochs'), gpus = dict_args.get('gpus'), 
                            callbacks = [StochasticWeightAveraging(swa_epoch_start = 0.7 , annealing_strategy = 'linear')], 
                            logger = logger, log_every_n_steps = 100)

    trainer.fit(model, train_loader, val_loader)

    #At the end, just removing some variables from memeory 
    del data
    del labels
    del ft
    del train_loader
    del val_loader