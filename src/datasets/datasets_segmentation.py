# Author: Anne Wannenwetsch, TU Darmstadt (anne.wannenwetsch@visinf.tu-darmstadt.de)
import os
import random

import numpy as np
import torch
import torch.utils.data

from datasets import data_utils


class SegmentationDataset(torch.utils.data.Dataset):
    """Loads a semantic segmentation dataset."""

    def __init__(self,
                 data_root,
                 data_list,
                 logits_root,
                 invalid_label,
                 transform=None):
        """Initializes the semantic segmentation dataset.
        Args:
            data_root: Directory with labels and input images.
            data_list: List of data samples.
            flow_root: Directory with input flow fields.
            transform: Transformation applied to all inputs.
        """
        self.data_list = data_utils.read_data_list(data_list)
        self.data_root = data_root
        self.logits_root = logits_root
        self.transform = transform
        self.invalid_label = invalid_label

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_names = self.data_list[index]
        image_path = os.path.join(self.data_root, data_names[0])
        image = data_utils.load_image(image_path)

        label_path = os.path.join(self.data_root, data_names[1])
        label = data_utils.load_segmentation(label_path, self.invalid_label)

        file_name = data_names[0][:-4]
        logits_path = os.path.join(self.logits_root, file_name + '.npy')
        logits = np.load(logits_path)

        if self.transform is not None:
            image, logits, label = self.transform(image, logits, label)

        label = data_utils.convert_data(label)
        logits = data_utils.convert_data(logits).float()
        image = data_utils.convert_data(image).float()

        return image, logits, label


class AugmenterSegmentation(object):
    """Normalizes and if applicable crops data."""

    def __init__(self, mean, std, crop_shape=None):
        """Initializes the segmentation augmenter.
        Args:
            mean: image mean value
            std: image standard deviation
            crop_shape: Size of cropped data.
        """
        self.mean = mean
        self.std = std
        self.crop_shape = crop_shape

    def __call__(self, image, logits, label):
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
            logits = logits[start_height:end_height, start_width:end_width, :]

        # Normalize input image.
        for i, (channel_mean, channel_std) in enumerate(
                zip(self.mean, self.std)):
            image[:, :, i] = (image[:, :, i] - channel_mean) / channel_std

        return image, logits, label
