"""Code for representing the KOI dataset."""
import os
import warnings
import pandas as pd
from lightkurve import search_targetpixelfile, LightkurveWarning

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


def download_all_short_cadence_observations_for_star_list(star_list):
    """Downloads and saves all the short cadence observations for the stars which appear in the star list."""
    observations_skipped = 0
    targets_skipped = 0
    for kepler_input_catalog_number in star_list:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=LightkurveWarning)  # Ignore warnings about empty downloads.
            current_target_skipped = False
            short_cadence_observations = search_targetpixelfile(kepler_input_catalog_number, cadence='short')
            for observation_index, observation in enumerate(short_cadence_observations):
                observation_file_name = f'{kepler_input_catalog_number}_{observation_index}.fits'
                try:
                    observation.download().to_fits(os.path.join(data_directory, observation_file_name), overwrite=True)
                except OSError:
                    observations_skipped += 1
                    if not current_target_skipped:
                        current_target_skipped = True
                        targets_skipped += 1
    print(f'{targets_skipped} targets skipped.')
    print(f'{observations_skipped} observations skipped.')


if __name__ == '__main__':
    star_list_ = get_easy_positive_stars()
    print_short_cadence_observation_count_for_stars(star_list_)
