import os
import re
import imageio
from datetime import datetime
import numpy as np


def main():

    path_to_images = '/Users/dnf/Downloads/him_gif_data/'
    filenames = np.array(os.listdir(path_to_images))
    times = np.array([int(re.search("[_][0-9]{4}[_]", f).group().replace("_", "")) for f in filenames])
    print times
    time_mask = (times > 1100) & (times < 2300)
    print time_mask
    filenames = filenames[~time_mask]
    print filenames

    with imageio.get_writer('/Users/dnf/Downloads/him_gif/mov.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(os.path.join(path_to_images, filename))
            writer.append_data(image)


if __name__ == "__main__":
    main()