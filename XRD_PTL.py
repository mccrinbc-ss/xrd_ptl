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

class XRD_dataset(ParentDataset):
    def __init__(self, label_location, XRD_location = None, K = 300, fourier_flag = False):
        
        self.fourier_flag = fourier_flag #bool 
        spectras = pd.read_feather(XRD_location) #spectras is an A x 2000 df of XRD spectras, clipped to > 0

        #Rewriting above
        print('Processing Batch...')
        # Load labels, grab features, filter.
        # features = ['t_b', 't_w', 's_b','s_w','num_loops','l_b','l_w']
        # labels = pd.read_csv('mtl_props.csv')
        
  ### Label / Material Props Processing 
        all_features = ['thickness_qw','thickness_qb','strain_qw','strain_qb',
                          'num_loops','wavelength_qw','wavelength_qb','net_strain',
                          'x_qw','y_qw','x_qb','y_qb','mismatch_arcsec','net_thickness','critical_thickness']

        features = ['thickness_qw', 'thickness_qb', 'strain_qw', 'strain_qb','num_loops', 'wavelength_qw', 'wavelength_qb']
        # labels = pd.read_feather(label_location)
        # labels = labels[features] #index columns that we're interested in

        #The labels now come from the data location:
        labels = spectras.T[-len(all_features):].T 
        labels.columns = all_features
        labels = labels[features] #We should be able to do the 

        spectras = spectras.T[0:-len(all_features)].T #Filtering the spectras.
        

        # Filter the datasets by some conditions you're interested in 
        labels = labels[(labels.num_loops >=30) & (labels.num_loops <=35)] # & 
            # ((labels.wavelength_qw - labels.wavelength_qb).abs() >= 200) & 
            # ((labels.wavelength_qw - labels.wavelength_qb).abs() <= 400)]

        # labels = labels[(labels.num_loops >=30) & (labels.num_loops <=35)]

        #Q11 Q14


        indicies_to_keep = labels.index 
        labels.reset_index(drop = True, inplace = True)
        print(f'Number of Samples After Filtering: {len(indicies_to_keep)}')
        
        scaler_labels = StandardScaler() #minmax scaling each column. Might need to be StandardScaler()
        labels = pd.DataFrame(scaler_labels.fit_transform(labels)) 

        self.labels  = labels
        self.scaler_labels = scaler_labels #if we wanted to unscale the label 

 ### Spectra Processing 
        spectras = spectras.iloc[indicies_to_keep] #filtered by the conditions set above. 
        spectras.reset_index(drop = True, inplace = True)

        spectras = spectras.clip(10**-9, 10**9)

 # ADD THIS BACK IN IF YOU WANT LOG NORMALIZATION. 
        #Conversion to numpy is done for speed
        vals = np.log10(spectras.values)
        vals = vals + np.abs(np.expand_dims(vals.min(axis = 1), axis = 0).T) + 10**-9 #add small value.
        sc_spectras = StandardScaler()
        spectras = sc_spectras.fit_transform(vals) #array of standard normalized spectra. 

        self.spectra = spectras #return the 150,000 x 2000 array 

 ### Fourier Processing
        if fourier_flag == True:
            print('Fourier Flag is True')

            InP_peak = np.load('InP_bragg.npy')
            ft_InP = rfft(InP_peak, workers = 4)

            gs = gaussian_filter1d(vals, sigma = 1) #gs is the gaussian smoothed signal. vals is already log10() and offset. 
            ft = rfft(gs, axis = 1, workers = 4)

            #Removing the InP Fourier Components from the spectra. 
            ft.real = ft.real - ft_InP.real
            ft.imag = ft.imag - ft_InP.imag

            ft = np.concatenate((ft[:, :K].real, ft[:, :K].imag), axis = 1) #concat the real and img components together

            ft_sc = StandardScaler() #first normalize the data the data on a per frequency basis
            ft = ft_sc.fit_transform(ft)

            # spec_normalizer = MinMaxScaler() #then min-max scale each spectrum
            # ft = spec_normalizer.fit_transform(ft.T).T #transpose, normalize, revert back. 

            # self.ft = df.values #this will be the size of 150,000 x 2*K #This is for min-max
            self.ft = ft

    def __getitem__(self, ii):
        mtl_props = self.labels.iloc[ii].values 
        label = torch.tensor(mtl_props).float() #convert to tensor 

        if self.fourier_flag == False:
            xrd = self.spectra[ii]
            xrd = torch.tensor(xrd).float().unsqueeze(0) #unsqueeze because signal needs to be [batch, 1, signal_length]
            return xrd, label

        elif self.fourier_flag == True:
            fourier_signal = self.ft[ii] #ft is a numpy array 
            fourier_signal = torch.tensor(fourier_signal).float() #linear model. fourier_signal is an array
            return fourier_signal, label
        
    def __len__(self):
        return len(self.labels) #Length of labels is the same as the number of XRD spectra

class AutoencoderXRD_newmodel(pl.LightningModule):
    def __init__(self, lr, dropout, momentum, recon_weight, batch_size, num_workers, **kwargs):
        
        self.lr            = lr               #set the learning rate.
        self.dropout       = dropout          #Regularization
        self.momentum      = momentum	      #BatchNorm Parameter
        self.alpha_recon   = recon_weight     #Weight placed on reconstruction 
        self.alpha_latent  = 1 - recon_weight
        self.batch_size    = batch_size
        self.num_workers   = num_workers      # > 0
        
        self.save_hyperparameters()
        
        super(AutoencoderXRD_newmodel, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8, stride=4, padding=2, bias = False), #output: size 500
            # nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias = False), #output: size 250
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, padding = 0), #reducing the size of the model 
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias = False),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, padding = 1), #reducing the size of the model 
            
            nn.Flatten(start_dim = 1, end_dim = 2),
            
            nn.Linear(in_features=128 * 32, out_features=256, bias = False), #16x reduction 
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            
            nn.Linear(in_features=256, out_features=7, bias = False),
            # nn.BatchNorm1d(7),
            # nn.Sigmoid()
            # nn.LeakyReLU()
            nn.Tanh()
        )
        
        self.decoder = nn.Sequential(
            
            nn.Linear(in_features=7, out_features=256, bias = False),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            
            nn.Linear(in_features=256, out_features=128 * 32, bias = False),
            # nn.BatchNorm1d(128 * 32),
            nn.LeakyReLU(),
            
            nn.Unflatten(-1, (128, 32)),
            nn.Upsample(size=62, mode = 'linear'), #increase the size of the input. 
            
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, stride = 2, padding=0, bias = False),
            # nn.BatchNorm1d(64), 
            nn.LeakyReLU(),
            nn.Upsample(size=250, mode = 'linear'),
            
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias = False),
            # nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            
            nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=8, stride=4, padding=2)
            # nn.BatchNorm1d(1)
            # nn.Sigmoid()
            
        )
    
    #PyTorch Lightning recommends that the training and inference portions of the loop are seperate. 
    def forward(self, x):
        embedding = self.encoder(x)
        return embedding
    
    def weighted_mse_loss(self, pred, target, weight = None):
        if weight is None:
            weight = 1
        return torch.mean(weight * (pred - target) ** 2) #weighting the mse loss by the ratio 
    
    
    #We can set an optimizer for the training loop. 
    def configure_optimizers(self):
        #print(self.lr)
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        scheduler_ = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', #min mode, lr reduced when quantity has stopped dec
                                                                 factor = 0.20, #factor to reduce lr by. lr = lr * factor
                                                                 patience = 5, #wait 5 epochs before reducing with no change.
                                                                 threshold = 0.0001,
                                                                 min_lr = 1e-8,
                                                                 verbose = True)
        # scheduler = {
        #     'scheduler': scheduler_,
        #     'reduce_on_plateau': True,
        #     # val_checkpoint_on is val_loss passed in as checkpoint_on
        #     #'monitor': 'val_checkpoint_on'
        #     'monitor' : 'val_loss'
        #             }
        
        return [optimizer]#, [scheduler]

    def training_step(self, batch, batch_idx):
        data, mtl_params = batch #input, label
        embedding = self.encoder(data) 
        x_prediction = self.decoder(embedding)

        recon_loss =  self.alpha_recon *F.mse_loss(x_prediction, data)
        latent_loss = self.alpha_latent*F.mse_loss(embedding, mtl_params)
        loss = recon_loss #+ latent_loss

        self.log('train_loss', loss) #logging into tensorboard for future analysis. 
        return loss
    
    #We have to remember that we care about this latent representation. 
    #Therefore, during validation, we compare the embedding vector with the LABEL. 
    def validation_step(self, val_batch, batch_idx):
        data, mtl_params = val_batch
        embedding = self.forward(data)
        
        #x_prediction = self.decoder(embedding)
        loss = F.mse_loss(embedding, mtl_params)
        
        #loss = F.mse_loss(embedding, mtl_params)
        self.log('val_loss', loss)
        
    def test_step(self, test_batch, batch_idx):
        data, data_noise, mtl_params  = test_batch
        recon = self.decoder(mtl_params)
        loss = F.mse_loss(recon, data) #loss between reconstruction and true signal
        self.log('test_loss', loss)

class AutoencoderXRD_oldmodel(pl.LightningModule):
    def __init__(self, lr, dropout, momentum, recon_weight, batch_size, num_workers, **kwargs):
        
        self.lr            = lr               #set the learning rate.
        self.dropout       = dropout          #Regularization
        self.momentum      = momentum	      #BatchNorm Parameter
        self.alpha_recon   = recon_weight     #Weight placed on reconstruction 
        self.alpha_latent  = 1 - recon_weight
        self.batch_size    = batch_size
        self.num_workers   = num_workers      # > 0
        
        self.save_hyperparameters()
        
        super(AutoencoderXRD_oldmodel, self).__init__()
        
        self.encoder = nn.Sequential(
        
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8, stride=8, padding=0, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p = self.dropout),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=5, padding=0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p = self.dropout),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=3, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p = self.dropout),
            
            nn.Flatten(start_dim = 1, end_dim = 2),
            
            nn.Linear(in_features=2048, out_features=128, bias = False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p = self.dropout),
            
            nn.Linear(in_features=128, out_features=7, bias = False), #latent representation,
            nn.BatchNorm1d(7)
            # nn.Sigmoid()
        
        )
        
        self.decoder = nn.Sequential(
            
            nn.Linear(in_features=7, out_features=128, bias = True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(in_features=128, out_features=2048, bias = True),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            
            nn.Unflatten(-1, (128, 16)),
            
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=5, stride = 3, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=5, stride=5, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=8, stride=8, padding=0),
            nn.BatchNorm1d(1)
            
        )
    
    #PyTorch Lightning recommends that the training and inference portions of the loop are seperate. 
    def forward(self, x):
        embedding = self.encoder(x)
        return embedding
    
    def weighted_mse_loss(self, pred, target, weight = None):
        if weight is None:
            weight = (target / pred).abs()
            weight = torch.log(1/pred)
        return torch.mean(weight * (pred - target) ** 2) #weighting the mse loss by the ratio 
    
    
    #We can set an optimizer for the training loop. 
    def configure_optimizers(self):
        #print(self.lr)
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        scheduler_ = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', #min mode, lr reduced when quantity has stopped dec
                                                                 factor = 0.20, #factor to reduce lr by. lr = lr * factor
                                                                 patience = 5, #wait 5 epochs before reducing with no change.
                                                                 threshold = 0.0001,
                                                                 min_lr = 1e-8,
                                                                 verbose = True)
        # scheduler = {
        #     'scheduler': scheduler_,
        #     'reduce_on_plateau': True,
        #     # val_checkpoint_on is val_loss passed in as checkpoint_on
        #     #'monitor': 'val_checkpoint_on'
        #     'monitor' : 'val_loss'
        #             }
        
        return [optimizer]#, [scheduler]

    def training_step(self, batch, batch_idx):
        data, mtl_params = batch #input, label
        embedding = self.encoder(data) 
        x_prediction = self.decoder(embedding)
        recon_loss =  self.alpha_recon *F.mse_loss(x_prediction, data, weight = 1)
        latent_loss = self.alpha_latent*F.mse_loss(embedding, mtl_params)
        # loss = recon_loss + latent_loss
        loss = recon_loss
        self.log('train_loss', loss) #logging into tensorboard for future analysis. 
        
        return loss
    
    #We have to remember that we care about this latent representation. 
    #Therefore, during validation, we compare the embedding vector with the LABEL. 
    def validation_step(self, val_batch, batch_idx):
        data, mtl_params = val_batch
        embedding = self.forward(data)
        loss = F.mse_loss(embedding, mtl_params)
        self.log('val_loss', loss)
        
    def test_step(self, test_batch, batch_idx):
        data, data_noise, mtl_params  = test_batch
        recon = self.decoder(mtl_params)
        loss = F.mse_loss(recon, data) #loss between reconstruction and true signal
        self.log('test_loss', loss)

class AutoencoderXRD_fourier(pl.LightningModule):

    def __init__(self, lr, dropout, momentum, recon_weight, batch_size, num_workers, max_epochs, K, **kwargs):
        
        self.lr            = lr               #set the learning rate.
        self.dropout       = dropout          #Regularization
        self.momentum      = momentum	      #BatchNorm Parameter
        self.alpha_recon   = recon_weight / 10     #Weight placed on reconstruction 
        self.alpha_latent  = 1 - (recon_weight / 10)
        self.batch_size    = batch_size
        self.num_workers   = num_workers      # > 0
        self.K             = K                #Number of fourier comps

        
        self.save_hyperparameters("recon_weight", "batch_size", "max_epochs")
        
        super(AutoencoderXRD_fourier, self).__init__()

        self.encoder = nn.Sequential( 
            nn.Linear(in_features = 2*K, out_features = 300, bias = True), 
            nn.ReLU(),

            nn.Linear(in_features = 300, out_features = 150, bias = True), 
            nn.ReLU(),

            nn.Linear(in_features = 150, out_features = 75, bias = True), 
            nn.BatchNorm1d(75),
            nn.ReLU(),

            nn.Linear(in_features=75, out_features=25, bias = True),
            nn.ReLU(),

            nn.Linear(in_features=25, out_features=7, bias = True), #input directly to latent representation
            # nn.BatchNorm1d(7)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=7, out_features = 25, bias = True),
            nn.ReLU(),

            nn.Linear(in_features=25, out_features = 75, bias = True),
            nn.ReLU(),

            nn.Linear(in_features=75, out_features = 150, bias = True),
            nn.BatchNorm1d(150),
            nn.ReLU(),

            nn.Linear(in_features=150, out_features = 300, bias = True),
            nn.ReLU(),

            nn.Linear(in_features=300, out_features = 2*K, bias = True),
            # nn.BatchNorm1d(2*K)

            )

	#PyTorch Lightning recommends that the training and inference portions of the loop are seperate. 
    def forward(self, x):
        embedding = self.encoder(x)
        return embedding
    
    # def weighted_mse_loss(self, pred, target, weight = None):
    #     if weight is None:
    #         weight = (target / pred).abs()
    #         weight = torch.log(1/pred)
    #     return torch.mean(weight * (pred - target) ** 2) #weighting the mse loss by the ratio 

    # def r2_score_torch(self, y_true, y_pred):
    #     numerator = ((y_true - y_pred) ** 2).sum(axis=0)
    #     denominator = ((y_true - torch.mean(y_true, axis=0)) ** 2).sum(axis=0)
    #     nonzero_denominator = denominator != 0
    #     nonzero_numerator = numerator != 0
    #     valid_score = (nonzero_denominator & nonzero_numerator).to('cuda')
    #     output_scores = torch.ones([y_true.shape[1]]).to('cuda')

    #     output_scores[valid_score] = 1 - (numerator[valid_score]/ denominator[valid_score])
    #     # # arbitrary set to zero to avoid -inf scores, having a constant
    #     # # y_true is not interesting for scoring a regression anyway
    #     output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0

    #     r2_score = torch.mean(output_scores)
    #     if r2_score < 0: #Way worse than we want. 
    #         r2_score = torch.tensor(0).float()

    #     return 1 - r2_score #loss is defined as LOW is good. 

    # def r2_score_torch_simp(self, y_pred, y_true):
    #     var_y = torch.var(y_true, unbiased=False)
    #     r2 = F.mse_loss(y_pred, y_true, reduction="mean") / var_y
    #     # if r2 < 0:
    #     #     r2.data = 1 - torch.tensor(0).float() #this is done to not break the loss function backprop connection
    #     return r2
    
    
    #We can set an optimizer for the training loop. 
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        scheduler_ = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', #min mode, lr reduced when quantity has stopped dec
                                                                 factor = 0.20, #factor to reduce lr by. lr = lr * factor
                                                                 patience = 5, #wait 5 epochs before reducing with no change.
                                                                 threshold = 0.0001,
                                                                 min_lr = 1e-8,
                                                                 verbose = True)
        # scheduler = {
        #     'scheduler': scheduler_,
        #     'reduce_on_plateau': False,
        #     # val_checkpoint_on is val_loss passed in as checkpoint_on
        #     #'monitor': 'val_checkpoint_on'
        #     'monitor' : 'val_loss'
        #             }
        
        return [optimizer]#, [scheduler]

    def training_step(self, batch, batch_idx):
        data, mtl_params = batch #input, label
        embedding = self.encoder(data) 
        x_prediction = self.decoder(embedding)
        # recon_loss =  self.alpha_recon*self.weighted_mse_loss(x_prediction, data, weight = 1)
        # latent_loss = self.alpha_latent*F.mse_loss(embedding, mtl_params)

        recon_loss =  F.mse_loss(x_prediction, data)

        # recon_loss =  self.r2_score_torch_simp(x_prediction, data)
        latent_loss = F.mse_loss(embedding, mtl_params)

        loss = recon_loss + latent_loss
        self.log('train_loss', loss) #logging into tensorboard for future analysis. 
        
        return loss
    
    #We have to remember that we care about this latent representation. 
    #Therefore, during validation, we compare the embedding vector with the LABEL. 
    def validation_step(self, val_batch, batch_idx):
        data, mtl_params = val_batch
        embedding = self.forward(data)
        loss = F.mse_loss(embedding, mtl_params)
        self.log('val_loss', loss)
        
    def test_step(self, test_batch, batch_idx):
        data, data_noise, mtl_params  = test_batch
        recon = self.decoder(mtl_params)
        loss = F.mse_loss(recon, data) #loss between reconstruction and true signal
        self.log('test_loss', loss)

class XRD_dataset_2(ParentDataset):
    def __init__(self, data, labels, ft, fourier_flag, single_sample = True, rep_factor = 100):
        '''
        data:   standardized,  np.array
        labels: standardized,  np.array
        ft: 0 or standardized, np.array
        '''

        if single_sample == True:
            index = 300 

            data = data[0:index, :] #take the 0th spectra
            labels  = labels[0:index, :]
            ft     = ft[0:index, :]
            print(f'Single Sample is True. Shape of ft array = {ft.shape}')
            # print(pd.DataFrame(label, columns = features).describe())
            
            data = np.tile(data,(rep_factor,1)) #tile 1 signal so we can train the data on this. 
            labels  = np.tile(labels, (rep_factor,1))
            ft     = np.tile(ft, (rep_factor,1))
        
        else:
            data = np.tile(data,(rep_factor,1)) #tile 1 signal so we can train the data on this. 
            labels  = np.tile(labels, (rep_factor,1))
            ft     = np.tile(ft, (rep_factor,1))

        print(f'Number of Samples after rep = {ft.shape}')
        self.spectra = data
        self.labels  = labels
        self.ft      = ft
        self.fourier_flag = fourier_flag
        
        
    def __getitem__(self, ii):
        mtl_props = self.labels[ii] 
        label = torch.tensor(mtl_props).float() #convert to tensor 

        if self.fourier_flag == False:
            xrd = self.spectra[ii]
            xrd = torch.tensor(xrd).float().unsqueeze(0) #unsqueeze because signal needs to be [batch, 1, signal_length]
            return xrd, label

        elif self.fourier_flag == True:
            fourier_signal = self.ft[ii] #ft is a numpy array 
            fourier_signal = torch.tensor(fourier_signal).float() #linear model. fourier_signal is an array
            return fourier_signal, label
        
    def __len__(self):
        return len(self.labels) #Length of labels is the same as the number of XRD spectra


def run_fourier(vals, K):
    '''
    Create a set of fourier comps when fed in an array of non-standardized logged spectra.
    '''
    InP_peak = np.load('InP_bragg.npy')
    ft_InP = rfft(InP_peak, workers = 4)

    gs = gaussian_filter1d(vals, sigma = 1) #gs is the gaussian smoothed signal. vals is already log10() and offset. 
    ft = rfft(gs, axis = 1, workers = 4)

    #Removing the InP Fourier Components from the spectra. 
    ft.real = ft.real - ft_InP.real
    ft.imag = ft.imag - ft_InP.imag

    ft = np.concatenate((ft[:, :K].real, ft[:, :K].imag), axis = 1) #concat the real and img components together
    return ft

def process_data(XRD_location, label_location = None, fourier_flag = False, K = 300, indicies_to_keep = False):
    spectras = pd.read_feather(XRD_location) #spectras is an A x 2000 df of XRD spectras, clipped to > 0

    #Rewriting above
    print('Processing Batch in New Function...')
    
 ### Label / Material Props Processing 
    all_features = ['thickness_qw','thickness_qb','strain_qw','strain_qb',
                        'num_loops','wavelength_qw','wavelength_qb','net_strain',
                        'x_qw','y_qw','x_qb','y_qb','mismatch_arcsec','net_thickness','critical_thickness']

    features = ['thickness_qw', 'thickness_qb', 'strain_qw', 'strain_qb','num_loops', 'wavelength_qw', 'wavelength_qb']

    #The labels now come from the data location:
    labels = spectras.T[-len(all_features):].T 
    labels.columns = all_features
    labels = labels[features]  

    spectras = spectras.T[0:-len(all_features)].T #Filtering the spectras.
    
    labels_sc = StandardScaler() #minmax scaling each column. Might need to be StandardScaler()
    labels = labels_sc.fit_transform(labels)

 ### Spectra Processing 
    spectras = spectras.clip(10**-11, 10**9)

 # ADD THIS BACK IN IF YOU WANT LOG NORMALIZATION. 
    #Conversion to numpy is done for speed
    vals = np.log10(spectras.values)
    vals = vals + np.abs(np.expand_dims(vals.min(axis = 1), axis = 0).T) + 10**-11 #add small value.
    spectras_sc = StandardScaler()
    spectras = spectras_sc.fit_transform(vals) #array of standard normalized spectra. 

 ### Fourier Processing
    if fourier_flag == True:
        print('Fourier Flag is True')

        ft = run_fourier(vals, K) #function 
        ft_sc = StandardScaler() #first normalize the data the data on a per frequency basis
        ft = ft_sc.fit_transform(ft)
    else:
        ft = np.array([0])


    #This is where we're going to be filtering the data based on some umap conditions: 
    if indicies_to_keep is not False:
        labels = labels[indicies_to_keep] #indicies_to_keep = Int64Index
        spectras = spectras[indicies_to_keep]

        if fourier_flag == True:
            ft = ft[indicies_to_keep]

        print(f'number of samples after umap: {spectras.shape}')

    return spectras, labels, ft, spectras_sc, labels_sc, ft_sc

        

def main(args):
    dict_args = vars(args)

    print('Recon Weight: ' + str(dict_args.get('recon_weight')/10.0))

    model_name = dict_args.get('model_name')
    label_dir = dict_args.get('label_dir')
    data_dir = dict_args.get('data_dir')
    K = dict_args.get('K') #number of fourier comps

    if args.model_name == "new_AE":
        model = AutoencoderXRD_newmodel(**dict_args)
        fourier_flag = False
    elif args.model_name == "old_AE":
        model = AutoencoderXRD_oldmodel(**dict_args)
        fourier_flag = False
    elif args.model_name == "fourier":
        model = AutoencoderXRD_fourier(**dict_args)
        fourier_flag = True
    else:
        print('No valid model selected.')

    model_name = f'{model_name}'
    logger = TensorBoardLogger("lightning_logs", name = model_name, default_hp_metric = False)

    data, labels, ft = process_data(data_dir, label_location = label_dir, fourier_flag = fourier_flag, K = K)
    datasets = XRD_dataset_2(data, labels, ft, fourier_flag)

    train_frac, val_frac, test_frac = (0.90, 0.10, 0.0)
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

    #This is where we actually train the model 
    #Automatically saves the best model checkpoint in a lightning logs folder. 
    trainer = pl.Trainer(max_epochs= dict_args.get('max_epochs'), gpus = dict_args.get('gpus'), 
                         callbacks = [StochasticWeightAveraging(swa_epoch_start = 0.7 , annealing_strategy = 'linear')], 
                         logger = logger)

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = ArgumentParser()

    #Create a set of arguments
    parser.add_argument("--model_name", type = str, default = "fourier", help="old_AE, new_AE, fourier")
    parser.add_argument("--data_dir", type = str, default = '/home/brian/projects/XRD_PTL/test_data/batch0_all_data.feather', help="Data directory")
    parser.add_argument("--label_dir", type = str, default = '/home/brian/projects/XRD_PTL/test/labels.feather', help="old_AE or new_AE")
    parser.add_argument("--K", type = int, default = 300, help = "Number of Fourier Comps")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.1)
    parser.add_argument("--recon_weight", type=float, default=6)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=1000)

    main(parser.parse_args())


#Personal notes:
# Larger batch sizes results in smoother losses, but should be trained for longer to achieve similar results than smaller batches. 
# Small model trained with many epochs, large batch size, seems to get reliable training performance. 
# Removing Batch normalization before latent smoothed validation performance. Removing BN at the end made no obvious difference for the simple linear model. 
# Fourier Models 16 + have a single 150 neuron hidden layer before the latent vector.

# Get an ensemble for uncertainty both also improved performance. 
# Don't forget to try different initialization 