from argparse import ArgumentParser

import numpy as np 
import pandas as pd

import xrayutilities as xu
from xrayutilities.materials import CubicAlloy, InP, GaP, InAs, GaAs

from tqdm import tqdm
from glob import glob

import re
import gc #Garbage collection 
import logging
import json 

logger = logging.getLogger('XRDLogger') #Creating a logger


class InGaAs(CubicAlloy):
    def __init__(self, x):
        """
        In_{1-x} Ga_x As cubic compound
        """
        super().__init__(InAs, GaAs, x)
        
class InGaP(CubicAlloy):
    def __init__(self, x):
        """
        In_{1-x} Ga_x P cubic compound
        """
        super().__init__(InP, GaP, x)

class InGaAsP(CubicAlloy):
    def __init__(self, x, y):
        """
        In_{1-x} Ga_x As_{1-y} P_y cubic compound
        """
        super().__init__(InGaAs(x), InGaP(x), y)
        self.name = 'Q'

def save_sims(Sims, sim_filepath):

    #Save all data as a dataframe for now. Might have to seperate into individual files. 
    xrds = pd.DataFrame(Sims) #index = np.arange(first_index, last_index)
    xrds.columns = list(map(str, xrds.columns.values.tolist()))#feathers need to have string column names. 

    xrds.to_feather(sim_filepath) #Save location is /scratch/k/kleiman/mccrinbc/data/sim_results/BATCH_NAME.feather    
    logger.info(f'Wrote chunk {sim_filepath}')

def simulate_xrd(mtl_params, cal_factor, sub, energy, resolution_width, ai):
    [t_wq, t_qb, num_loops, x_qw, y_qw, x_qb, y_qb] = mtl_params

    barrier = InGaAsP(1-x_qb, y_qb)
    well    = InGaAsP(1-x_qw, y_qw)
    
    lay_barrier = xu.simpack.Layer(barrier, t_qb*cal_factor, roughness=0.0, relaxation=0.) #barrier 
    lay_well = xu.simpack.Layer(well, t_wq*cal_factor, roughness=0.0, relaxation=0.) #well

    #Creation of the device we're interested in 
    pstack = xu.simpack.PseudomorphicStack001('device', 
                                              sub + lay_barrier + int(num_loops)*(lay_well + lay_barrier)
                                             )

    dyn  = xu.simpack.DynamicalModel(pstack, 
                                     energy=energy, 
                                     resolution_width=resolution_width,
                                    )

    Idyn = dyn.simulate(ai, hkl=(0, 0, 4))
    return Idyn

def main(args):
    '''
    Script to generate XRD Spectra from terminal input. 

    Written: Brian McCrindle
    '''

    dict_args = vars(args) #read in the params

    #Calibration factor needs to be set to correct for thickness from Q_Calculator 
    cal_factor = dict_args.get('cal_factor')
    energy = 'CuKa1'# eV
    resolution_width = 0.004 #35
    bragg_angle = 31.6682
    om_2_th = np.linspace(-5000, 5000, 10000//5) #generating 2000 samples per XRD curve
    ai = om_2_th/3600 + bragg_angle

    BATCH_SIZE = dict_args.get('batch_size')
    INDEX_BEGIN = dict_args.get('index_begin')
    CHUNK_SIZE = dict_args.get('chunk_size')

    input_path = glob(dict_args.get('data_path'))[0] #The datapath has a * at the end and we need to search for this file. 

    df = pd.read_feather(input_path) #create the dataframe to loop over
    batch_feather_string = re.search('batch.*', input_path)[0] #name is the batch number that identifies the data path with a .feather extention. 

    logger_filename = dict_args.get('save_path') + '/' + batch_feather_string
    logging.basicConfig(format='%(name)s::%(asctime)s::%(levelname)s::%(message)s',
                                    filename=logger_filename.replace('.feather','.log'), 
                                    # filemode='w', # this flag creates a new file each time
                                    level=logging.DEBUG)

    logging.debug('Running Script with arguments: \n %s', json.dumps(dict_args, indent = 2))   

    s = InP
    sub = xu.simpack.Layer(s, float('inf'), roughness=0.0, relaxation=0.)

    Sims = [] #storing all simulations 
    garb = 0

    #Generating XRD simulations for a 32*(Q11, Q14) structure
    mtl_params = ['thickness_qw', 'thickness_qb', 'num_loops', 'x_qw', 'y_qw', 'x_qb', 'y_qb']

    try:
        print('getting into Try')
        for ii in range(BATCH_SIZE):

            mtl_info = df.iloc[INDEX_BEGIN + ii]
            Idyn = simulate_xrd(mtl_info[mtl_params], cal_factor, sub, energy, resolution_width, ai) #pass in the material parameters that are needed for device construction. return the simulation (Idyn). 
            
            #Extend the array so that we include the material properties. 
            Idyn = np.concatenate((Idyn, mtl_info.values)) #ALL the material props in the feather file, not just the ones we use for the simulation. 
            
            Sims.append(Idyn)

            first_index = INDEX_BEGIN + ii - CHUNK_SIZE
            last_index  = INDEX_BEGIN + ii

            if (ii > 0) and (ii % CHUNK_SIZE == 0): #We're processing the data in batches to avoid memory overload.

                sim_filepath = dict_args.get('save_path') + '/' + batch_feather_string.replace('.', f'_chunk{INDEX_BEGIN + ii - CHUNK_SIZE}_{INDEX_BEGIN + ii}.')
                save_sims(Sims, sim_filepath) #First and last index are being used column renaming so the df is labelled properly. 

                del Sims #Attemping to avoid memory leakage. 

                gc.collect() #Garbage collection. Works even if declared to a variable name.
                Sims = []
    except: 
        print('getting into except')
        #We may have run simulations but the chunk starts to index a simulation number > 149999. Save these simulations!
        logging.debug('Outside of the indexable dataframe. INDEX_BEGIN + ii == ' + str(INDEX_BEGIN + ii)) 
        sim_filepath = dict_args.get('save_path') + '/' + batch_feather_string.replace('.', f'_chunk{INDEX_BEGIN + ii - CHUNK_SIZE}_{INDEX_BEGIN + ii}.')
        save_sims(Sims, sim_filepath)

        del Sims
        Sims = []


    if len(Sims) > 0:        
        sim_filepath = dict_args.get('save_path') + '/' + batch_feather_string.replace('.', f'_chunk{INDEX_BEGIN + ii - CHUNK_SIZE}_{INDEX_BEGIN + ii}.')
        save_sims(Sims, sim_filepath)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_path", type = str, default = "/Users/brianmccrindle/Documents/SolidStateAI/Niagara_Cluster/material_params_below_2ct/material_params_below2ct_batch0_*", help="Location of mtl feather file")
    parser.add_argument("--save_path", type = str, default = "/Users/brianmccrindle/Documents/SolidStateAI/Niagara_Cluster/test_sims", help="Save folder location of XRD feather file")
    parser.add_argument("--batch_size", type = int, default = 100, help="Number of XRD Spectra to Run")
    parser.add_argument("--index_begin", type = int, default = 0, help="Index of first XRD spectra to generate")
    parser.add_argument("--chunk_size", type = int, default = 0, help="number of spectra to save in each feather file")
    parser.add_argument("--cal_factor", type = float, default = 0.76, help="Thickness Calibration Factor (Float)")

    main(parser.parse_args())




