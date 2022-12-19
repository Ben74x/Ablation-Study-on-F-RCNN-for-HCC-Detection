# Reference: https://github.com/johschmidt42/PyTorch-Object-Detection-Faster-RCNN-Tutorial/blob/master/anchor_script.ipynb

# imports
import pathlib

import numpy as np
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from pytorch_faster_rcnn.datasets import ObjectDetectionDataSet
from pytorch_faster_rcnn.transformations import Clip, ComposeDouble
from pytorch_faster_rcnn.transformations import FunctionWrapperDouble
from pytorch_faster_rcnn.transformations import normalize_01
from pytorch_faster_rcnn.utils import get_filenames_of_path
from pytorch_faster_rcnn.visual import AnchorViewer


# root directory
root = pathlib.Path('pytorch_faster_rcnn_tutorial/Dataset')

# input and target files
inputs = get_filenames_of_path(root / 'input')
targets = get_filenames_of_path(root / 'target')

# Sort files
inputs.sort()
targets.sort()

# mapping
mapping = {
    'HCC': 1,
    'Non-HCC': 2
}


# transforms
transforms = ComposeDouble([
    Clip(),
    # AlbumentationWrapper(albumentation=A.HorizontalFlip(p=0.5)),
    # AlbumentationWrapper(albumentation=A.RandomScale(p=0.5, scale_limit=0.5)),
    # AlbumentationWrapper(albumentation=A.VerticalFlip(p=0.5)),
    FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])


# dataset building
dataset = ObjectDetectionDataSet(inputs=inputs,
                                 targets=targets,
                                 transform=transforms,
                                 use_cache=False,
                                 convert_to_format=None,
                                 mapping=mapping)

# transforms
transform = GeneralizedRCNNTransform(min_size=1024,
                                     max_size=1024,
                                     image_mean=[0.485, 0.456, 0.406],
                                     image_std=[0.229, 0.224, 0.225])


image = dataset[0]['x']  # ObjectDetectionDataSet
feature_map_size = (512, 32, 32)

anchorviewer = AnchorViewer(image=image,
                            rcnn_transform=transform,
                            feature_map_size=feature_map_size,
                            anchor_size=((128, 256, 512),),
                            aspect_ratios=((0.5, 1.0, 2.0),)
                            )
anchorviewer.napari()