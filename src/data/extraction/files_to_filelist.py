'''
Gets a list of files in a folder and writes it
to a text file.

File list is used for telling ORAC which files to process
'''

import os
import src.config.filepaths as filepaths

with open(os.path.join(filepaths.path_to_filelists, 'indonesia_filepaths.txt'), 'wb') as text_file:
    for f in os.listdir(filepaths.path_to_modis_l1b):
        print f
        text_file.write(f + '\n')

