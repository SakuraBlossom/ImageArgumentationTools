import os
import sys
import copy
import argparse

from PIL import Image
from xml.dom import minidom

def get_args():
    parser = argparse.ArgumentParser(description='Something smart here')
    parser.add_argument("-t", dest='target_folder', help='Path to target folder', nargs='*', required=True)
    return parser.parse_args()


def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, str(percents).rjust(5), '%', suffix))
    sys.stdout.flush()

list_of_folders = get_args().target_folder
print(list_of_folders)
for idx_folder, folder_dir in enumerate(list_of_folders, start=1):
    full_path = os.path.abspath(folder_dir)
    list_of_files = os.listdir(full_path)

    for idx_file, filename in enumerate(list_of_files):
        full_filename = full_path + '/' + filename
        progress(idx_file, len(list_of_files), \
            suffix='[{} of {}] Scanning {}'.format(idx_folder, len(list_of_folders), folder_dir))
        if filename.endswith('.jpg') or filename.endswith('.png'):
            try:
                img = Image.open(full_filename) # open the image file
                img.verify() # verify that it is, in fact an image
            except (IOError, SyntaxError) as e:
                print('Bad file: {}'.format(filename)) # print out the names of corrupt files
        if filename.endswith('.xml'):
            try:
                mydoc = minidom.parse(full_filename)
            except (IOError, SyntaxError) as e:
                print('Bad file: {}'.format(filename)) # print out the names of corrupt files
    print('')

print('DONE')