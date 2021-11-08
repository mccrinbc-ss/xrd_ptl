import os 
import numpy as np
from scipy.fft import rfft, irfft, fftfreq
from scipy.ndimage import gaussian_filter1d

class fourier_tools():

    def __init__(self):
        '''
        Set of fourier-based tools to process XRD spectra. 
        Assumptions:
            - InP frequency comps are to be removed from all spectra.
        '''

        self.ep = 10**11 #offset value.
        
        InP_spectra = np.load('InP_bragg.npy') #load in the InP XRD spectra
        df_vals = np.log10(InP_spectra)
        InP_norm = df_vals + np.abs(np.expand_dims(df_vals.min(), axis = 0).T) + self.ep #To avoid 0s in the log10

        InP_ft, gs = self.fourier(InP_norm, InP = True)

        self.InP_real = InP_ft.real
        self.InP_imag = InP_ft.imag
        

    def fourier(self, all_data, InP = False):
        '''
        Fourier transform of an array of values. 
        all_data: df of all spectra (row basis) to be processed
        '''

        all_data = all_data.clip(self.ep, 10**9)

        df_vals = np.log10(all_data.values)
        min_vals = np.abs(np.expand_dims(df_vals.min(axis = 1), axis = 0).T) #NEED THIS FOR THE FOURIER RECOSNTRUCTION
        df_vals_offset = df_vals + min_vals + self.ep # abs([1, min_value_of(150000_rows)].T ) + ep to avoid 0

        gs = gaussian_filter1d(df_vals_offset, sigma = 1) #gs is the gaussian smoothed signal 
        if InP:
            axis_ = 0
        else: 
            axis_ = 1
        ft = rfft(gs, axis = axis_, workers = 4)

        self.min_vals = min_vals #we need this if we're looking to use fourier_to_XRD


        return ft, gs #fourier df

    def fourier_to_XRD(self, ft, sc, K = 300):
        '''
        Function to convert a set of fourier comps into it's correspodning XRD signal 
        ft: fourier signal [real, imag]. Expected as np.array
        sc: StandardScaler used to transform fourier data.
        ep: offset
        min_vals: 
        '''
        ft_non_scaled = sc.inverse_transform(ft)
        comps = ft_non_scaled[:, :K] + 1j *ft_non_scaled[:, K:]

        #Adding back in the InP fourier components so we get a normal looking signal back. 
        comps.real = comps.real + self.InP_real[:K]
        comps.imag = comps.imag + self.InP_imag[:K]

        recon_signals_original = irfft(comps, n=2000)

        signals = recon_signals_original - self.ep - self.min_vals

        base = np.ones(signals.shape) * 10 #everything here is in base 10
        signals = np.power(base, signals)

        return signals