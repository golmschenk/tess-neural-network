"""Code for downloading the Kepler flare catalog data."""
import os
import warnings

import numpy as np
from lightkurve import search_targetpixelfile, LightkurveWarning

from flare_catalog_dataset import data_directory, catalog_path


def get_count_stars_in_flare_catalog_with_short_cadence_data():
    """Returns the number of targets in the flare catalog with short cadence data."""
    flare_catalog = np.genfromtxt(catalog_path, delimiter=',', skip_header=1)
    kepler_input_catalog_numbers = [target[0] for target in flare_catalog]
    count_with_short_cadence = 0
    for kepler_input_catalog_number in kepler_input_catalog_numbers:
        short_cadence_observations = search_targetpixelfile(kepler_input_catalog_number, cadence='short')
        if len(short_cadence_observations) is not 0:
            count_with_short_cadence += 1
    return count_with_short_cadence


def download_all_short_cadence_observations_for_targets_in_flare_catalog():
    """Downloads and saves all the short cadence observations for the stars which appear in the flare catalog."""
    flare_catalog = np.genfromtxt(catalog_path, delimiter=',', skip_header=1)
    kepler_input_catalog_numbers = list(map(int, [target[0] for target in flare_catalog]))
    observations_skipped = 0
    targets_skipped = 0
    for kepler_input_catalog_number in kepler_input_catalog_numbers:
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
    download_all_short_cadence_observations_for_targets_in_flare_catalog()
