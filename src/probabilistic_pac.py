# Author: Anne Wannenwetsch, TU Darmstadt (anne.wannenwetsch@visinf.tu-darmstadt.de)
# Parts of this code were adapted from https://github.com/NVlabs/pacnet
import copy
import torch

import pac


class NormalizedPacConv2d(pac.PacConv2d):
    """Implements a pixel-adaptive convolution with advanced normalization."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 bias=True,
                 kernel_type='gaussian',
                 shared_filters=False):
        """Initializes PAC with advanced normalization.
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Filter size of used kernel.
            padding: Number of zero padding elements applied at all borders.
            bias: Usage of bias term.
            kernel_type: Type of kernel function K. See original PAC for
                available options.
            shared_filters: Sharing of filters among input dimensions.
        """
        super(NormalizedPacConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=1,
            bias=bias,
            kernel_type=kernel_type,
            smooth_kernel_type='none',
            normalize_kernel=False,
            shared_filters=shared_filters,
            filler='uniform',
            native_impl=False)
        # Create normalization weight.
        self.weight_normalization = torch.nn.parameter.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        # Initialize convolution and normalization weight with same positive
        # values.
        self.weight.data = torch.abs(self.weight.data)
        self.weight_normalization.data = torch.log(
            torch.abs(copy.deepcopy(self.weight.data)))

    def forward(self, input_features, input_for_kernel, kernel=None,
                mask=None):
        """Returns pixel-adaptive convolution with advanced normalization."""

        # Compute pixel-adaptive kernel.
        output_mask = None
        if kernel is None:
            kernel, output_mask = self.compute_kernel(input_for_kernel, mask)

        # Perform pixel-adaptive convolution.
        channels = input_features.shape[1]
        output = pac.pacconv2d(input_features, kernel, self.weight, None,
                               self.stride, self.padding, self.dilation,
                               self.shared_filters, self.native_impl)

        # Determine normalization factor dependent on kernel and weight.
        if self.shared_filters:
            normalization_factor = torch.einsum(
                'ijklmn,zykl->ijmn',
                (kernel, torch.exp(self.weight_normalization)))
        else:

            normalization_factor = torch.einsum(
                'ijklmn,ojkl->iomn', (kernel.repeat(1, channels, 1, 1, 1, 1),
                                      torch.exp(self.weight_normalization)))
        # Crop normalization factor for numerical stability.
        normalization_factor = torch.max(normalization_factor,
                                         torch.tensor([1e-20]).cuda())
        output = output / normalization_factor

        # Bias term added after normalization.
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)

        return output if output_mask is None else (output, output_mask)


class ProbPacConv2d(pac.PacConv2d):
    """Implements a probabilistic pixel-adaptive convolution."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 bias=True,
                 kernel_type='gaussian',
                 shared_filters=False):
        """Initializes PPAC.
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Filter size of used kernel.
            padding: Number of zero padding elements applied at all borders.
            bias: Usage of bias term.
            kernel_type: Type of kernel function K. See original PAC for
                available options.
            shared_filters: Sharing of filters among input dimensions.
        """
        super(ProbPacConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=1,
            bias=bias,
            kernel_type=kernel_type,
            smooth_kernel_type='none',
            normalize_kernel=False,
            shared_filters=shared_filters,
            filler='uniform',
            native_impl=False)
        # Create normalization weight.
        self.weight_normalization = torch.nn.parameter.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        # Initialize convolution and normalization weight with same positive
        # values.
        self.weight.data = torch.abs(self.weight.data)
        self.weight_normalization.data = torch.log(
            torch.abs(copy.deepcopy(self.weight.data)))

    def forward(self,
                input_features,
                input_for_probabilities,
                input_for_kernel,
                kernel=None,
                mask=None):
        """Returns result of probabilistic pixel-adaptive convolution."""

        # Compute pixel-adaptive kernel.
        output_mask = None
        if kernel is None:
            kernel, output_mask = self.compute_kernel(input_for_kernel, mask)

        # Multiply input with probabilities and perform standard PAC.
        batch_size, channels = input_features.shape[:2]
        input_features = input_features * input_for_probabilities
        output = pac.pacconv2d(input_features, kernel, self.weight, None,
                               self.stride, self.padding, self.dilation,
                               self.shared_filters, self.native_impl)

        # Determine normalization factor dependent on probabilities, kernel and
        # convolution weight.
        neighbor_probabilities = torch.nn.functional.unfold(
            input_for_probabilities, self.kernel_size, self.dilation,
            self.padding, self.stride)
        neighbor_factors = neighbor_probabilities.view(
            batch_size, 1, *kernel.shape[2:]) * kernel
        if self.shared_filters:
            normalization_factor = torch.einsum(
                'ijklmn,zykl->ijmn',
                (neighbor_factors, torch.exp(self.weight_normalization)))
        else:
            normalization_factor = torch.einsum(
                'ijklmn,ojkl->iomn',
                (neighbor_factors.repeat(1, channels, 1, 1, 1, 1),
                 torch.exp(self.weight_normalization)))
        # Crop normalization factor for numerical stability.
        normalization_factor = torch.max(normalization_factor,
                                         torch.tensor([1e-20]).cuda())
        output = output / normalization_factor

        # Bias term added after normalization.
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)

        return output if output_mask is None else (output, output_mask)
