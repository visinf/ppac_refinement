# Author: Anne Wannenwetsch, TU Darmstadt (anne.wannenwetsch@visinf.tu-darmstadt.de)
import os
import random

import numpy as np
import torch
import torch.utils.data

from data import hd3data
from datasets import data_utils


class FlowDataset(torch.utils.data.Dataset):
    """Loads an optical flow dataset."""

    def __init__(self, data_root, data_list, flow_root, transform=None):
        """Initializes the optical flow dataset.
        Args:
            data_root: Directory with labels(, masks) and input images.
            data_list: List of data samples.
            flow_root: Directory with input flow fields.
            transform: Transformation applied to all inputs.
        """
        self.data_list = data_utils.read_data_list(data_list)
        self.data_root = data_root
        self.flow_root = flow_root
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_names = self.data_list[index]

        # Load label and masks if required.
        label_path = os.path.join(self.data_root, data_names[2])
        label = hd3data.read_gen(label_path, "flow")
        if len(data_names) == 4:
            mask_path = os.path.join(self.data_root, data_names[3])
            mask = hd3data.read_gen(mask_path, "mask")
            label = np.concatenate((label, mask), 2)

        # Load first input image.
        image_path = os.path.join(self.data_root, data_names[0])
        image = data_utils.load_image(image_path)

        # Load input flow field.
        file_name = data_names[0][:-4]
        flow_path = os.path.join(self.flow_root, file_name + '.npy')
        flow = np.load(flow_path)

        # Load input probability map.
        prob_path = os.path.join(self.flow_root, file_name + '_prob.npy')
        prob = np.load(prob_path)

        # Transform inputs if required.
        if self.transform is not None:
            image, flow, prob, label = self.transform(image, flow, prob, label)

        label = data_utils.convert_data(label).float()
        flow = data_utils.convert_data(flow).float()
        prob = data_utils.convert_data(prob).float()
        image = data_utils.convert_data(image).float()
        return (image, flow, prob, label)


class AugmenterFlow(object):
    """Normalizes and if applicable crops data."""

    def __init__(self, mean, std, crop_shape=None):
        """Initializes the flow augmenter.
        Args:
            mean: image mean value
            std: image standard deviation
            crop_shape: Size of cropped data.
        """
        self.mean = mean
        self.std = std
        self.crop_shape = crop_shape

    def __call__(self, image, flow, prob, label):
        # Crop input data if required.
        if self.crop_shape is not None:
            label_shape = label.shape[:2]
            if label_shape < self.crop_shape:
                raise Exception("Label too small for given crop_shape.")

            start_height = random.randint(0,
                                          label_shape[0] - self.crop_shape[0])
            end_height = start_height + self.crop_shape[0]
            start_width = random.randint(0,
                                         label_shape[1] - self.crop_shape[1])
            end_width = start_width + self.crop_shape[1]
            label = label[start_height:end_height, start_width:end_width, :]
            image = image[start_height:end_height, start_width:end_width, :]
            flow = flow[start_height:end_height, start_width:end_width, :]
            prob = prob[start_height:end_height, start_width:end_width, :]

        # Normalize input image.
        for i, (channel_mean, channel_std) in enumerate(
                zip(self.mean, self.std)):
            image[:, :, i] = (image[:, :, i] - channel_mean) / channel_std

        return image, flow, prob, label
