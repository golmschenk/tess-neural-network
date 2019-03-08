"""Code for representing the KOI dataset."""
import os
import numpy as np
import pandas as pd
from lightkurve import search_targetpixelfile

data_directory = 'koi_data'
catalog_path = os.path.join(data_directory, 'cumulative_2019.03.08_12.09.57.csv')

def get_easy_positive_stars():
    """Gets star IDs which have confirmed exoplanets with short periods"""
    koi_data_frame = pd.read_csv(catalog_path, usecols=['kepid', 'koi_disposition', 'koi_period'])
    confirmed_data_frame = koi_data_frame.loc[koi_data_frame['koi_disposition'] == 'CONFIRMED']
    short_period_data_frame = confirmed_data_frame.loc[confirmed_data_frame['koi_period'] <= 10]
    easy_positive_stars = short_period_data_frame['kepid'].unique()
    return easy_positive_stars

def print_short_cadence_observation_count_for_stars(star_list):
    """Print the number of stars and observations with short cadence in the given star list."""
    count_of_stars_with_short_cadence = 0
    count_of_observations_with_short_cadence = 0
    for kepler_input_catalog_number in star_list:
        short_cadence_observations = search_targetpixelfile(kepler_input_catalog_number, cadence='short')
        count_of_observations_with_short_cadence += len(short_cadence_observations)
        if len(short_cadence_observations) is not 0:
            count_of_stars_with_short_cadence += 1
    print(f'Stars: {count_of_stars_with_short_cadence}')
    print(f'Observations: {count_of_observations_with_short_cadence}')
