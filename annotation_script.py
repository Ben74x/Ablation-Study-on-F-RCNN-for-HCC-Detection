# Reference: https://github.com/johschmidt42/PyTorch-Object-Detection-Faster-RCNN-Tutorial/blob/master/annotation_script.ipynb

# imports
import os
import pathlib


# Annotations
from pytorch_faster_rcnn.utils import get_filenames_of_path
from pytorch_faster_rcnn.visual import Annotator

directory = pathlib.Path('Indiv_Proj-main/Dataset/')
image_files = get_filenames_of_path(directory / 'input')

annotator = Annotator(image_ids=image_files)
annotator.napari()


# Add labels
annotator.add_class(label='HCC', color='red') # HCC
#annotator.add_class(label='Non-HCC', color='yellow') # Non-HCC


# Save annotations
save_dir = pathlib.Path(os.getcwd()) / 'targets'
save_dir.mkdir(parents=True, exist_ok=True)
annotator.export_all(pathlib.Path(save_dir))