# Author: Anne Wannenwetsch, TU Darmstadt (anne.wannenwetsch@visinf.tu-darmstadt.de)
import math

import torch


def safe_log(x, eps=1e-20):
    """Performs numerically stable log operation."""
    x = torch.max(x, torch.tensor([eps]).cuda())
    return torch.log(x)


def get_upsampled_probabilities_hd3(full_vect, full_prob):
    """Upsamples and interpolates HD3 probabilities."""
    levels = len(full_prob)
    batch_size, _, H_est, W_est = full_vect[0].shape
    corr_range = int(math.sqrt(full_prob[0].size(1))) // 2
    upsampled_probabilities = torch.zeros(batch_size, levels, H_est,
                                          W_est).cuda()

    previous_flow = torch.zeros(batch_size, 2, H_est, W_est).cuda()
    for level in range(levels):
        # Upsample and normalize current probability estimate.
        current_prob = full_prob[level]
        H_prob, W_prob = current_prob.shape[2:]
        current_prob = torch.nn.functional.interpolate(
            current_prob, (H_est, W_est), mode='bilinear', align_corners=True)
        current_prob = torch.nn.functional.softmax(current_prob, dim=1)

        # Compute flow residual and rescale w.r.t to original probability size.
        current_flow = full_vect[level]
        current_residual = current_flow - previous_flow
        current_residual = torch.stack([
            current_residual[:, 0, :, :] * float(W_prob / W_est),
            current_residual[:, 1, :, :] * float(H_prob / H_est)
        ],
                                       dim=1)

        # Interpolate grid probabilities using nearest neighbors.
        upsampled_probabilities[:, level:level +
                                1, :, :] = interpolate_probabilities(
                                    current_residual, current_prob, corr_range)

        previous_flow = current_flow

    return upsampled_probabilities


def interpolate_probabilities(flow, prob, corr_range):
    """Performs nearest neighbor interpolation of probabilities prob."""
    x_coordinates = torch.clamp(
        flow[:, 0, :, :], min=-corr_range, max=corr_range)
    y_coordinates = torch.clamp(
        flow[:, 1, :, :], min=-corr_range, max=corr_range)
    x_nearest = torch.round(x_coordinates)
    y_nearest = torch.round(y_coordinates)
    index = (y_nearest + corr_range) * (
        2 * corr_range + 1) + x_nearest + corr_range
    nearest_prob = torch.gather(prob, 1, index.unsqueeze(1).long())
    return nearest_prob
