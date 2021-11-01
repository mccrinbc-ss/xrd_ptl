from glob import glob
import pandas as pd
from argparse import ArgumentParser


def main(args):
    dict_args = vars(args)

    data_directory = dict_args.get('data_dir')
    save_directory = dict_args.get('save_dir')
    batch_num = dict_args.get('batch_num')

    PATH = data_directory + f'/batch{batch_num}*.feather' #get all the file locations that have batch{batch_num}, ignore log files
    PATHS = glob(PATH)

    all_data = pd.DataFrame()
    for path in PATHS: 
        data = pd.read_feather(path)
        all_data = all_data.append(data)

    all_data.reset_index(drop = True, inplace = True) #need to do this because feather is dumb
    all_data.to_feather(save_directory + f'/batch{batch_num}_all_data.feather')


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type = str, default = "/scratch/k/kleiman/mccrinbc/data/new_sims", help="Location of mtl feather files")
    parser.add_argument("--save_dir", type = str, default = "/scratch/k/kleiman/mccrinbc/data/agg_data_2", help="Save location")
    parser.add_argument("--batch_num", type = int, default = 0)

    main(parser.parse_args())