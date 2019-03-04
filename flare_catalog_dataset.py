"""Code for representing the flare catalog dataset."""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import lightkurve

data_directory = 'data'
catalog_path = os.path.join(data_directory, 'flare_catalog.csv')


class FlareCatalogDataset(Dataset):
    """A class to represent the flare catalog dataset."""
    def __init__(self):
        self.catalog_data_frame = pd.read_csv(catalog_path, index_col=0)
        self.observation_file_names = [file_name for file_name in os.listdir(data_directory) if
                                       file_name.endswith('.fits')]

    def __len__(self):
        return len(self.observation_file_names)

    def __getitem__(self, index):
        observation_file_name = self.observation_file_names[index]
        observation = lightkurve.open(os.path.join(data_directory, observation_file_name))
        target_flare_data = self.catalog_data_frame.loc[observation.targetid]
        flare_frequency_coefficients = (target_flare_data['alpha_ffd'], target_flare_data['beta_ffd'])
        return torch.tensor(observation.flux.newbyteorder()), torch.tensor(flare_frequency_coefficients)

if __name__ == '__main__':
    flare_catalog_dataset = FlareCatalogDataset()
    x = flare_catalog_dataset[0]
    print(x)
