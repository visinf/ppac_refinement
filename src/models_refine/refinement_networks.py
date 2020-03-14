# Author: Anne Wannenwetsch, TU Darmstadt (anne.wannenwetsch@visinf.tu-darmstadt.de)
import numpy as np
import torch
import torch.nn as nn

import probabilistic_pac


class PPacNet(nn.Module):
    """Implements architecture for probabilistic pixel-adaptive refinement."""

    def __init__(self,
                 kernel_size_preprocessing,
                 kernel_size_joint,
                 conv_specification,
                 shared_filters,
                 depth_layers_prob,
                 depth_layers_guidance,
                 depth_layers_joint,
                 fixed_gaussian_weights=False,
                 bias_zero_init=False):
        super(PPacNet, self).__init__()

        # Determine and verify setup for ppacs.
        self.conv_specification = conv_specification
        self.depth_layers_joint = depth_layers_joint
        num_pac = conv_specification.count('p')
        assert depth_layers_prob[-1] == num_pac
        self.guidance_share = int(depth_layers_guidance[-1] / num_pac)
        assert self.guidance_share * num_pac == depth_layers_guidance[-1]

        # Setup guidance network.
        layers_guidance = []
        for i in range(1, len(depth_layers_guidance)):
            layers_guidance.append(
                nn.Conv2d(
                    depth_layers_guidance[i - 1],
                    depth_layers_guidance[i],
                    kernel_size=kernel_size_preprocessing,
                    stride=1,
                    padding=kernel_size_preprocessing // 2,
                    dilation=1,
                    bias=True))
            if i < len(depth_layers_guidance) - 1:
                layers_guidance.append(nn.ReLU(inplace=True))
        self.network_guidance = nn.Sequential(*layers_guidance)

        # Setup probability network.
        layers_prob = []
        for i in range(1, len(depth_layers_prob)):
            layers_prob.append(
                nn.Conv2d(
                    depth_layers_prob[i - 1],
                    depth_layers_prob[i],
                    kernel_size=kernel_size_preprocessing,
                    stride=1,
                    padding=kernel_size_preprocessing // 2,
                    dilation=1,
                    bias=True))
            if i < len(depth_layers_prob) - 1:
                layers_prob.append(nn.ReLU(inplace=True))
        self.network_prob = nn.Sequential(*layers_prob)

        # Setup combination network.
        layers_joint = nn.ModuleList()
        for i, conv_type in enumerate(conv_specification):
            if conv_type == 'p':
                layers_joint.append(
                    probabilistic_pac.ProbPacConv2d(
                        depth_layers_joint[i],
                        depth_layers_joint[i + 1],
                        kernel_size_joint,
                        padding=kernel_size_joint // 2,
                        bias=True,
                        kernel_type='gaussian',
                        shared_filters=shared_filters))
            elif conv_type == 'c':
                layers_joint.append(
                    nn.Conv2d(
                        depth_layers_joint[i],
                        depth_layers_joint[i + 1],
                        kernel_size=kernel_size_joint,
                        stride=1,
                        padding=kernel_size_joint // 2,
                        dilation=1,
                        bias=True))
            else:
                raise ValueError('Unknown convolution type {}'.format(type))
        self.layers_joint = layers_joint

        # Initialize bias terms of combination branch with zeros if required.
        if bias_zero_init:
            for layer in self.layers_joint:
                torch.nn.init.zeros_(layer.bias)

        # Fix combination weights to Gaussians and bias to zero if required.
        if fixed_gaussian_weights:
            for layer in self.layers_joint:
                layer.weight.requires_grad = False
                layer.weight.data = gaussian_kernel(
                    kernel_size=kernel_size_joint, sigma=1.)
                layer.weight_normalization.data = torch.log(
                    gaussian_kernel(kernel_size=kernel_size_joint, sigma=1.))
                layer.weight_normalization.requires_grad = False
                layer.bias.requires_grad = False

    def forward(self, estimates, probabilities, images):
        """Returns the estimates refined by the PPAC network."""
        probabilities = self.network_prob(probabilities)
        probabilities = torch.sigmoid(probabilities)
        guidance = self.network_guidance(images)

        for i, conv_type in enumerate(self.conv_specification):
            if conv_type == 'p':
                pac_counter = self.conv_specification[:i].count('p')
                current_probabilities = probabilities[:, pac_counter:(
                    pac_counter + 1)]
                current_guidance = guidance[:, pac_counter * self.
                                            guidance_share:(pac_counter + 1) *
                                            self.guidance_share]
                estimates = self.layers_joint[i](
                    estimates, current_probabilities, current_guidance)
            elif conv_type == 'c':
                estimates = self.layers_joint[i](estimates)
        return estimates


class PacNetAdvancedNormalization(nn.Module):
    """Implements a sample architecture for pixel-adaptive refinement using PACs
    with advanced normalization scheme."""

    def __init__(self, kernel_size_preprocessing, kernel_size_joint,
                 conv_specification, shared_filters, depth_layers_guidance,
                 depth_layers_joint):
        super(PacNetAdvancedNormalization, self).__init__()

        # Determine and verify setup for pacs.
        self.conv_specification = conv_specification
        self.depth_layers_joint = depth_layers_joint
        num_pac = conv_specification.count('p')
        self.guidance_share = int(depth_layers_guidance[-1] / num_pac)
        assert self.guidance_share * num_pac == depth_layers_guidance[-1]

        # Setup guidance network.
        layers_guidance = []
        for i in range(1, len(depth_layers_guidance)):
            layers_guidance.append(
                nn.Conv2d(
                    depth_layers_guidance[i - 1],
                    depth_layers_guidance[i],
                    kernel_size=kernel_size_preprocessing,
                    stride=1,
                    padding=kernel_size_preprocessing // 2,
                    dilation=1,
                    bias=True))
            if i < len(depth_layers_guidance) - 1:
                layers_guidance.append(nn.ReLU(inplace=True))
        self.network_guidance = nn.Sequential(*layers_guidance)

        # Setup combination network.
        layers_joint = nn.ModuleList()
        for i, conv_type in enumerate(conv_specification):
            if conv_type == 'p':
                layers_joint.append(
                    probabilistic_pac.NormalizedPacConv2d(
                        depth_layers_joint[i],
                        depth_layers_joint[i + 1],
                        kernel_size_joint,
                        padding=kernel_size_joint // 2,
                        bias=True,
                        kernel_type='gaussian',
                        shared_filters=shared_filters))
            elif conv_type == 'c':
                layers_joint.append(
                    nn.Conv2d(
                        depth_layers_joint[i],
                        depth_layers_joint[i + 1],
                        kernel_size=kernel_size_joint,
                        stride=1,
                        padding=kernel_size_joint // 2,
                        dilation=1,
                        bias=True))
            else:
                raise ValueError('Unknown convolution type {}'.format(type))
        self.layers_joint = layers_joint

    def forward(self, estimates, images):
        """Returns the estimates refined by the PAC network."""
        guidance = self.network_guidance(images)
        for i, conv_type in enumerate(self.conv_specification):
            if conv_type == 'p':
                pac_counter = self.conv_specification[:i].count('p')
                current_guidance = guidance[:, pac_counter * self.
                                            guidance_share:(pac_counter + 1) *
                                            self.guidance_share]
                estimates = self.layers_joint[i](estimates, current_guidance)
            elif conv_type == 'c':
                estimates = self.layers_joint[i](estimates)
        return estimates


def gaussian_kernel(kernel_size=5, sigma=1.0):
    """Returns a Gaussian filter kernel."""
    kernel_range = (kernel_size - 1) / 2.
    distance = np.linspace(-kernel_range, kernel_range, kernel_size)
    distance_x, distance_y = np.meshgrid(distance, distance)
    squared_distance = distance_x**2 + distance_y**2
    gauss_kernel = np.exp(-0.5 * squared_distance / np.square(sigma))
    gauss_kernel = gauss_kernel / np.sum(gauss_kernel)
    return torch.tensor(gauss_kernel).float().unsqueeze(0).unsqueeze(0).cuda()
