# Author: Anne Wannenwetsch, TU Darmstadt (anne.wannenwetsch@visinf.tu-darmstadt.de)
import numpy as np
from PIL import Image
import torch


def read_data_list(data_list):
    """Reads and splits list of several data inputs."""
    with open(data_list) as list_file:
        data_names = list_file.read().splitlines()
    return [data.strip().split(' ') for data in data_names]


def load_image(image_path):
    """Loads and converts image."""
    image = Image.open(image_path)
    return np.asarray(image) / 255.


def load_segmentation(segmentation_path, invalid_label):
    """Loads segmentation map."""
    segmentation = Image.open(segmentation_path)
    segmentation = np.array(segmentation, dtype=np.long)
    segmentation[segmentation == invalid_label] = -1
    return np.expand_dims(segmentation, axis=2)


def convert_data(data):
    """Converts data from numpy array to torch tensor."""
    data = np.transpose(data, (2, 0, 1))
    return torch.from_numpy(data)


def create_colormap(dataset_name):
    """Creates color map for specific dataset."""
    if dataset_name == 'Pascal':
        return _create_pascal_colormap()
    else:
        raise ValueError('Unknown dataset name {}'.format(dataset_name))


def _create_pascal_colormap():
    colormap = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                         [0, 0, 128], [128, 0, 128], [0, 128, 128],
                         [128, 128, 128], [64, 0, 0], [192, 0, 0],
                         [64, 128, 0], [192, 128, 0], [64, 0, 128],
                         [192, 0, 128], [64, 128, 128], [192, 128, 128],
                         [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                         [0, 64, 128], [224, 224, 192]])
    return colormap
