

# rename downloaded files

# Reference: https://github.com/johschmidt42/PyTorch-Object-Detection-Faster-RCNN-Tutorial/blob/master/rename_files_script.ipynb

import pathlib

from utils import get_filenames_of_path

root = pathlib.Path('Indiv_Proj-main/Dataset/')

inputs = get_filenames_of_path(root / 'input')
inputs.sort()


for idx, path in enumerate(inputs):
    old_name = path.stem
    old_extension = path.suffix
    dir = path.parent
    new_name = str(idx).zfill(4) + old_extension
    print(f'old: {old_name + old_extension}')
    print(f'new: {new_name}')
    path.rename(pathlib.Path(dir, new_name))