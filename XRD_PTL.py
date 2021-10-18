from argparse import ArgumentParser
# from tqdm import tqdm

import pytorch_lightning as pl 
# from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np
# import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset as ParentDataset
from torch.utils.data import DataLoader, random_split

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

pl.utilities.seed.seed_everything(seed = 3654) #3564, 5555
pd.options.mode.chained_assignment = None

class XRD_dataset(ParentDataset):
    def __init__(self, label_location, XRD_location = None, fourier_location = None):

        if fourier_location is not None:
        	fcomps = pd.read_csv(fourier_location,  sep = ",", header = None) #Load signals. Already normalized. 
        	self.fcomps  = fcomps #we're assuming that we don't need to do any additional preprocessing to the data. 
        elif XRD_location is not None:
        	# spectra = pd.read_csv(XRD_location,  sep = ",", header = None) #Load signals.
            spectra = pd.read_feather(XRD_location)
	        # normalization scheme for XRD spectra
            for ii in range(len(spectra)):
                s = np.log10(spectra.iloc[ii]) 
                s = s + np.abs(s.min())  #move the signal to > 0
                s.fillna(0, inplace = True) #There are errors in the spectra causing NaNs when log10'd. Fill the NaNs with 0
                spectra.iloc[ii] = s #override the location of the signal. s == pd.Series

            spectra = ((spectra - spectra.min()) / (spectra.max() - spectra.min())).fillna(0) #min-max scaling
            self.spectra = spectra
       	 # spectra = ((spectra - spectra.mean())/spectra.std()).fillna(0) #standardization 
        
        # Load labels, grab features, filter.
        # features = ['t_b', 't_w', 's_b','s_w','num_loops','l_b','l_w']
        features = ['thickness_qw', 'thickness_qb', 'strain_qw', 'strain_qb','num_loops', 'wavelength_qw', 'wavelength_qb']
        # labels  = pd.read_csv(label_location, sep = ",")
        labels = pd.read_feather(label_location)
        labels = labels[features] #index columns that we're interested in
        
        scaler_labels = MinMaxScaler() #minmax scaling each column
        labels = pd.DataFrame(scaler_labels.fit_transform(labels)) 

        self.labels  = labels
        self.scaler_labels = scaler_labels #if we wanted to unscale the label 

    def __getitem__(self, ii):
        mtl_props = self.labels.iloc[ii].values 
        label = torch.tensor(mtl_props).float() #convert to tensor 

        if hasattr(self, 'fcomps'): #If we have the attribute
            fcomp = self.fcomps.iloc[ii].values
            fcomp = torch.tensor(fcomp).float() #if we're using fcomp, it's a linear model. therefore, no unsqueeze(0)
            return fcomp, label

        elif hasattr(self, 'spectra'): #If we have the attribute
            xrd = self.spectra.iloc[ii].values
            xrd = torch.tensor(xrd).float().unsqueeze(0) #unsqueeze because signal needs to be [batch, 1, signal_length]
            return xrd, label
        
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
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias = False), #output: size 250
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, padding = 0), #reducing the size of the model 
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias = False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, padding = 1), #reducing the size of the model 
            
            nn.Flatten(start_dim = 1, end_dim = 2),
            
            nn.Linear(in_features=128 * 32, out_features=256, bias = False), #16x reduction 
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(in_features=256, out_features=7, bias = False),
            nn.BatchNorm1d(7),
            nn.Sigmoid()
        )
        
        self.decoder = nn.Sequential(
            
            nn.Linear(in_features=7, out_features=256, bias = False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(in_features=256, out_features=128 * 32, bias = False),
            nn.BatchNorm1d(128 * 32),
            nn.ReLU(),
            
            nn.Unflatten(-1, (128, 32)),
            nn.Upsample(size=62, mode = 'linear'), #increase the size of the input. 
            
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, stride = 2, padding=0, bias = False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Upsample(size=250, mode = 'linear'),
            
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias = False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(1)
            
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
        scheduler = {
            'scheduler': scheduler_,
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            #'monitor': 'val_checkpoint_on'
            'monitor' : 'val_loss'
                    }
        
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        data, mtl_params = batch #input, label
        embedding = self.encoder(data) 
        x_prediction = self.decoder(embedding)

        recon_loss =  self.alpha_recon*self.weighted_mse_loss(x_prediction, data, weight = 1)
        latent_loss = self.alpha_latent*F.mse_loss(embedding, mtl_params)
        loss = recon_loss + latent_loss

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
            nn.Dropout(p),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=5, padding=0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=3, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p),
            
            nn.Flatten(start_dim = 1, end_dim = 2),
            
            nn.Linear(in_features=2048, out_features=128, bias = False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p),
            
            nn.Linear(in_features=128, out_features=7, bias = False), #latent representation,
            nn.BatchNorm1d(7),
            nn.Sigmoid()
        
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
        scheduler = {
            'scheduler': scheduler_,
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            #'monitor': 'val_checkpoint_on'
            'monitor' : 'val_loss'
                    }
        
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        data, mtl_params = batch #input, label
        embedding = self.encoder(data) 
        x_prediction = self.decoder(embedding)
        recon_loss =  self.alpha_recon*self.weighted_mse_loss(x_prediction, data, weight = 1)
        latent_loss = self.alpha_latent*F.mse_loss(embedding, mtl_params)
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

class AutoencoderXRD_fourier(pl.LightningModule):

    def __init__(self, lr, dropout, momentum, recon_weight, batch_size, num_workers, **kwargs):
        
        self.lr            = lr               #set the learning rate.
        self.dropout       = dropout          #Regularization
        self.momentum      = momentum	      #BatchNorm Parameter
        self.alpha_recon   = recon_weight     #Weight placed on reconstruction 
        self.alpha_latent  = 1 - recon_weight
        self.batch_size    = batch_size
        self.num_workers   = num_workers      # > 0
        
        self.save_hyperparameters()
        
        super(AutoencoderXRD_fourier, self).__init__()

        self.encoder = nn.Sequential( 
            nn.Linear(in_features=200, out_features=50, bias = False),
            nn.BatchNorm1d(50),
            nn.ReLU(),

            nn.Linear(in_features=50, out_features=7, bias = False), #latent representations
            nn.BatchNorm1d(7),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=7, out_features=50, bias = False),
            nn.BatchNorm1d(50),
            nn.ReLU(),

            nn.Linear(in_features=50, out_features=200, bias = False),
            nn.BatchNorm1d(200)

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
        scheduler = {
            'scheduler': scheduler_,
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            #'monitor': 'val_checkpoint_on'
            'monitor' : 'val_loss'
                    }
        
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        data, mtl_params = batch #input, label
        embedding = self.encoder(data) 
        x_prediction = self.decoder(embedding)
        recon_loss =  self.alpha_recon*self.weighted_mse_loss(x_prediction, data, weight = 1)
        latent_loss = self.alpha_latent*F.mse_loss(embedding, mtl_params)
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

def main(args):
	dict_args = vars(args)

	if (dict_args.get('model_name') == 'old_AE') or (dict_args.get('model_name') == 'new_AE'):
		datasets = XRD_dataset(label_location = dict_args.get('label_dir'), XRD_location = dict_args.get('data_dir'))
	else:
		datasets = XRD_dataset(label_location = dict_args.get('label_dir'), fourier_location = dict_args.get('fourier_dir'))

	train_frac, val_frac, test_frac = (0.70, 0.30, 0.0)
	total_samples = len(datasets) #use a length in the future. 

	train_num = int(total_samples * train_frac)
	val_num = int(total_samples * val_frac)
	# test_num = total_samples - train_num - val_num

	train_set, val_set, test_set = random_split(datasets, [train_num, val_num, test_num]) #this will only work for single batches being fed in

	# Create the dataloaders needed for PyTorch
	BS = dict_args.get('batch_size')
	NW = dict_args.get('num_workers')

	train_loader = DataLoader(train_set, BS, shuffle = True, num_workers = NW)
	val_loader = DataLoader(val_set, BS, shuffle = False, num_workers = NW)
	# test_loader = DataLoader(test_set, BS, shuffle = False, num_workers = NW)

	if args.model_name == "new_AE":
		model = AutoencoderXRD_newmodel(**dict_args)
	elif args.model_name == "old_AE":
		model = AutoencoderXRD_oldmodel(**dict_args)
	elif args.model_name == "fourier":
		model = AutoencoderXRD_fourier(**dict_args)
	else:
		print('No valid model selected.')

	#This is where we actually train the model 
	#Automatically saves the best model checkpoint in a lightning logs folder. 
	trainer = pl.Trainer(max_epochs= dict_args.get('max_epochs'), gpus = dict_args.get('gpus'), stochastic_weight_avg=True)
	trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
	parser = ArgumentParser()

	#Create a set of arguments
	parser.add_argument("--model_name", type = str, default = "new_AE", help="old_AE, new_AE, fourier")
	parser.add_argument("--data_dir", type = str, default = '/Users/brianmccrindle/Documents/SolidStateAI/Niagara_Cluster/batch37_data/batch37_sims_sorted.feather', help="Data directory")
	parser.add_argument("--label_dir", type = str, default = '/Users/brianmccrindle/Documents/SolidStateAI/Niagara_Cluster/material_params_below_2ct/material_params_below2ct_batch37_5550000-5700000.feather', help="old_AE or new_AE")
	parser.add_argument("--fourier_dir", type = str, default = None, help="Enter a location of fourier data.")
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--num_workers", type=int, default=1)
	parser.add_argument("--lr", type=float, default=0.001)
	parser.add_argument("--dropout", type=float, default=0.2)
	parser.add_argument("--momentum", type=float, default=0.1)
	parser.add_argument("--recon_weight", type=float, default=0.5)
	parser.add_argument("--gpus", type=int, default=0)
	parser.add_argument("--max_epochs", type=int, default=50)

	main(parser.parse_args())


#  XRD location 
# '/Users/brianmccrindle/Documents/SolidStateAI/XRayUtils/Generated_XRD_Data/XRD_Spectra.csv'
	
#Fourier Location 
#'/Users/brianmccrindle/Documents/SolidStateAI/XRayUtils/Generated_XRD_Data/fcomps.csv'










