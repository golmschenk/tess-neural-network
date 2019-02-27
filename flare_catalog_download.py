"""Code for downloading the Kepler flare catalog data."""
import os
import numpy as np
from lightkurve import search_targetpixelfile


def get_count_stars_in_flare_catalog_with_short_cadence_data():
    """Returns the number of targets in the flare catalog with short cadence data."""
    flare_catalog = np.genfromtxt(os.path.join('data', 'flare_catalog.csv'), delimiter=',', names=True)
    kepler_input_catalog_numbers = [target[0] for target in flare_catalog]
    count_with_short_cadence = 0
    for kepler_input_catalog_number in kepler_input_catalog_numbers:
        short_cadence_observations = search_targetpixelfile(kepler_input_catalog_number, cadence='short')
        if len(short_cadence_observations) is not 0:
            count_with_short_cadence += 1
    return count_with_short_cadence


if __name__ == '__main__':
    print(get_count_stars_in_flare_catalog_with_short_cadence_data())
