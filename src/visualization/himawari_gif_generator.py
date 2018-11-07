import os
import re
import imageio
from datetime import datetime
import numpy as np


def main():

    path_to_images = '/Users/dnf/Downloads/him_gif_data/'
    filenames = np.array(os.listdir(path_to_images))

    # first restrict to only valid data
    times = np.array([int(re.search("[_][0-9]{4}[_]", f).group().replace("_", "")) for f in filenames])
    time_mask = (times > 1100) & (times < 2300)
    filenames = filenames[~time_mask]

    # next get time subset
    min_time = datetime(2015, 9, 23)
    max_time = datetime(2015, 9, 30)
    times = [datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", f).group(), '%Y%m%d_%H%M') for f in filenames]
    time_mask = [min_time < t < max_time for t in times]
    filenames = filenames[time_mask]
    print filenames

    with imageio.get_writer('/Users/dnf/Downloads/him_gif/mov.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(os.path.join(path_to_images, filename))
            writer.append_data(image)


if __name__ == "__main__":
    main()