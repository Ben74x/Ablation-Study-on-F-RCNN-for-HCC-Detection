# Ablation Study On Faster RCNN For HCC Detection
The codebase for my postgraduate thesis can be found in this repository. The common type of liver cancer, hepatoma, has been thoroughly studied using the Faster RCNN model.

This project's main objective is to evaluate the performance (speed and accuracy) of Faster RCNN model on the liver cancer detection with various backbones and parameters.

---

## Installation

[conda](https://docs.conda.io/en/latest/miniconda.html):
- `conda create --name faster-rcnn-proj -y`
- `conda activate faster-rcnn-proj`
- `conda install python=3.8 -y`
        
[venv](https://docs.python.org/3/library/venv.html):
- `python3 -m venv faster-rcnn-proj`
- `source faster-rcnn-proj/bin/activate`



Install the libraries:
   `pip install -r requirements.txt`

**Note**: CPU-version of torch will be installed. If you want to use a GPU or TPU, please refer to the instructions
on the [PyTorch website](https://pytorch.org/). To check whether pytorch uses the nvidia gpu, check
if `torch.cuda.is_available()` returns `True` in your python shell.
   

---


## Dependencies

These are the libraries that are used in this project:

- High-level deep learning library for PyTorch: [PyTorch Lightning](https://www.pytorchlightning.ai/)
- Visualization software: Custom code with the image-viewer [Napari](https://napari.org/)
- [OPTIONAL] Experiment tracking software/logging module: [Neptune](https://neptune.ai/)

If you want to use [Neptune](https://neptune.ai/) for your own experiments, add the `NEPTUNE` environment variable to
your system. Before you do that, create an account and an api key on `NEPTUNE`. Otherwise, deactivate it in the scripts. 

## Dataset

The dataset consists of 1200 hand labelled images from [The Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080617#61080617bcab02c187174a288dbcbf95d26179e8). The labelling was done using [Napari](https://napari.org/) and details can be found in the annotation script. My labelled data can be found [here](https://drive.google.com/drive/folders/1d7WuqRQSmFah4tBh8Eo2eG-6v3nVvC8K?usp=sharing).

---


NVIDIA GeForce RTX 2080
