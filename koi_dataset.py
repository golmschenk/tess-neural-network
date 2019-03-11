"""Code for representing the KOI dataset."""
import math
import os
import random
import shutil
import warnings
from dataclasses import dataclass
import lightkurve
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from lightkurve import search_lightcurvefile, LightkurveWarning, search_targetpixelfile
from torch.utils.data import Dataset

data_directory = 'koi_data'
koi_catalog_path = os.path.join(data_directory, 'koi.csv')
kic_catalog_path = os.path.join(data_directory, 'kic.csv')
positive_data_directory = os.path.join(data_directory, 'positive')
negative_data_directory = os.path.join(data_directory, 'negative')
padded_example_length = 50000


@dataclass
class KoiExample:
    """A data class to represent the example information."""
    file_name: str
    label: bool


class KoiCatalogDataset(Dataset):
    """A class to represent the KOI catalog dataset."""
    def __init__(self, start=None, end=None, random_seed=0):
        positive_file_names = [file_name for file_name in os.listdir(positive_data_directory) if
                               file_name.endswith('.fits')]
        positive_examples = [KoiExample(file_name=file_name, label=True) for file_name in positive_file_names]
        negative_file_names = [file_name for file_name in os.listdir(negative_data_directory) if
                               file_name.endswith('.fits')]
        negative_examples = [KoiExample(file_name=file_name, label=False) for file_name in negative_file_names]
        all_examples = positive_examples + negative_examples
        random.seed(random_seed)  # Make sure the shuffling is consistent between train and test datasets.
        random.shuffle(all_examples)
        self.examples = all_examples[start:end]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        light_curve_file_name = example.file_name
        if example.label:
            example_directory = positive_data_directory
        else:
            example_directory = negative_data_directory
        light_curve = lightkurve.open(os.path.join(example_directory, light_curve_file_name))
        flux = light_curve.hdu[1].columns['FLUX'].array
        flux = flux.byteswap().newbyteorder()
        flux = (flux - flux.min()) / (flux.max() - flux.min())  # Normalize flux.
        padding_required = padded_example_length - flux.size
        if padding_required < 0:
            padded_flux = flux[:padding_required]
        else:
            padded_flux = np.pad(flux, (0, padding_required), mode='reflect')
        return torch.tensor(padded_flux), torch.tensor(example.label, dtype=torch.float32)


def get_easy_positive_stars():
    """Gets star IDs which have confirmed exoplanets with short periods."""
    koi_data_frame = pd.read_csv(koi_catalog_path, usecols=['kepid', 'koi_disposition', 'koi_period'])
    confirmed_data_frame = koi_data_frame.loc[koi_data_frame['koi_disposition'] == 'CONFIRMED']
    short_period_data_frame = confirmed_data_frame.loc[confirmed_data_frame['koi_period'] <= 10]
    easy_positive_stars = short_period_data_frame['kepid'].unique()
    return easy_positive_stars


def get_negative_stars():
    """Gets star IDs which are Kepler targets, but are not in the KOI table."""
    koi_data_frame = pd.read_csv(koi_catalog_path, usecols=['kepid'])
    random.seed(0)
    iter_csv = pd.read_csv(kic_catalog_path, sep='|', usecols=['kic_kepler_id', 'kic_fov_flag'], iterator=True,
                           chunksize=1000)
    kic_data_frame = pd.concat([chunk[chunk['kic_fov_flag'] == 2] for chunk in iter_csv])
    kic_not_in_koi_data_frame = kic_data_frame[~kic_data_frame['kic_kepler_id'].isin(koi_data_frame['kepid'])]
    star_list = kic_not_in_koi_data_frame['kic_kepler_id'].unique()
    random.shuffle(star_list)
    return star_list


def print_short_cadence_observation_count_for_stars(star_list):
    """Print the number of stars and observations with short cadence in the given star list."""
    count_of_stars_with_short_cadence = 0
    count_of_observations_with_short_cadence = 0
    for kepler_input_catalog_number in star_list:
        short_cadence_observations = search_lightcurvefile(kepler_input_catalog_number, cadence='short')
        count_of_observations_with_short_cadence += len(short_cadence_observations)
        if len(short_cadence_observations) is not 0:
            count_of_stars_with_short_cadence += 1
    print(f'Stars: {count_of_stars_with_short_cadence}')
    print(f'Observations: {count_of_observations_with_short_cadence}')


def download_all_short_cadence_observations_for_star_list(star_list, directory, max_observations=math.inf):
    """Downloads and saves all the short cadence observations for the stars which appear in the star list."""
    lightkurve_cache_path = os.path.join(str(Path.home()), '.lightkurve-cache')
    if os.path.exists(lightkurve_cache_path):
        shutil.rmtree(lightkurve_cache_path)  # Clear cache to prevent corrupted file downloads.
    observations_downloaded = 0
    stars_downloaded = 0
    os.makedirs(directory, exist_ok=True)
    for kepler_input_catalog_number in star_list:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=LightkurveWarning)  # Ignore warnings about empty downloads.
            short_cadence_observations = search_targetpixelfile(kepler_input_catalog_number, cadence='short')
            if len(short_cadence_observations) > 0:
                stars_downloaded += 1
            for observation_index, observation in enumerate(short_cadence_observations):
                observation_file_name = f'{kepler_input_catalog_number}_{observation_index}.fits'
                target_pixel_file = observation.download()
                light_curve = target_pixel_file.to_lightcurve(aperture_mask=target_pixel_file.pipeline_mask)
                light_curve.to_fits(os.path.join(directory, observation_file_name), overwrite=True)
                observations_downloaded += 1
                if observations_downloaded >= max_observations:
                    break
                print(f'\r{observations_downloaded} observations downloaded...', end='')
        if observations_downloaded >= max_observations:
            break
    print(f'{stars_downloaded} stars downloaded.')
    print(f'{observations_downloaded} observations downloaded.')


def remove_already_downloaded_stars_from_list(star_list, directory):
    """
    Removes stars which have already been downloaded from the list.
    Assumes the same list as the previous download is being used, in the same order, and that if a star was
    downloaded, it was fully downloaded.
    Best to avoid using this function, but as the MAST download can be unreliable, it allows us to finish downloading.
    """
    os.makedirs(directory, exist_ok=True)
    start_index_for_undownloaded = None
    for index, star in enumerate(reversed(star_list)):
        if any([file_name for file_name in os.listdir(directory) if file_name.startswith(str(star))]):
            start_index_for_undownloaded = -index
            print(f'Found star {start_index_for_undownloaded} (from end of list) already downloaded.')
            break
    return star_list[start_index_for_undownloaded:]


def print_dataset_statistics():
    """Prints off various statistics about the dataset."""
    dataset = KoiCatalogDataset()
    element_counts = []
    max_values = []
    min_values = []
    min_to_max_differences = []
    min_to_max_ratio = []
    for example in dataset.examples:
        light_curve_file_name = example.file_name
        if example.label:
            example_directory = positive_data_directory
        else:
            example_directory = negative_data_directory
        light_curve = lightkurve.open(os.path.join(example_directory, light_curve_file_name))
        flux = light_curve.hdu[1].columns['FLUX'].array
        element_counts.append(len(flux))
        max_values.append(flux.max())
        min_values.append(flux.min())
        min_to_max_differences.append(flux.max() - flux.min())
        min_to_max_ratio.append(flux.min() / flux.max())
    print(f'Max elements: {np.max(element_counts)}')
    print(f'Min elements: {np.min(element_counts)}')
    print(f'Std elements: {np.std(element_counts)}')
    print(f'Mean elements: {np.mean(element_counts)}')
    print(f'Max ratio: {np.max(min_to_max_ratio)}')
    print(f'Min ratio: {np.min(min_to_max_ratio)}')
    print(f'Std ratio: {np.std(min_to_max_ratio)}')
    print(f'Mean ratio: {np.mean(min_to_max_ratio)}')


if __name__ == '__main__':
    if not os.path.exists('koi_data/kic.csv') or not os.path.exists('koi_data/koi.csv'):
        print('Before running this script, add the CSV files noted in `koi_data/readme.md` to `koi_data`.')
        exit()
    print('Downloading positive observations...')
    positive_star_list = get_easy_positive_stars()
    positive_star_list = remove_already_downloaded_stars_from_list(positive_star_list, positive_data_directory)
    download_all_short_cadence_observations_for_star_list(positive_star_list, positive_data_directory)
    print('Downloading negative observations...')
    negative_star_list = get_negative_stars()
    negative_star_list = remove_already_downloaded_stars_from_list(negative_star_list, negative_data_directory)
    positive_observation_file_list = [file_name for file_name in os.listdir(positive_data_directory)
                                      if file_name.endswith('.fits')]
    existing_negative_observation_file_list = [file_name for file_name in os.listdir(negative_data_directory)
                                               if file_name.endswith('.fits')]
    observations_to_download = len(positive_observation_file_list) - len(existing_negative_observation_file_list)
    download_all_short_cadence_observations_for_star_list(negative_star_list, negative_data_directory,
                                                          max_observations=observations_to_download)
