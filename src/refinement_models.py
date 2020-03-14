# Author: Anne Wannenwetsch, TU Darmstadt (anne.wannenwetsch@visinf.tu-darmstadt.de)
# Parts of this code were adapted from https://github.com/ucbdrive/hd3
import torch.nn as nn

import losses


class EpeNet(nn.Module):
    """Implements model setup for optical flow refinement."""

    def __init__(self, refinement_net):
        super(EpeNet, self).__init__()
        self.criterion = losses.endpoint_error
        self.eval_outliers = losses.outlier_rate
        self.refinement_net = refinement_net

    def forward(self,
                flow,
                probabilities,
                images,
                label_list=None,
                get_flow=True,
                get_loss=False,
                get_epe=False,
                get_outliers=False):
        """Returns the output of a forward pass for flow refinement."""
        flow = self.refinement_net(flow, probabilities, images)
        result = {}
        if get_flow:
            result['flow'] = flow
        if get_loss:
            result['loss'] = self.criterion(flow, label_list[0])
        if get_epe:
            result['epe'] = self.criterion(flow, label_list[0])
        if get_outliers:
            result['outliers'] = self.eval_outliers(flow, label_list[0])
        return result


class CrossEntropyNet(nn.Module):
    """Implements model setup for semantic segmentation refinement."""

    def __init__(self, refinement_net):
        super(CrossEntropyNet, self).__init__()
        self.criterion = losses.CrossEntropySegmentationCalculator()
        self.refinement_net = refinement_net

    def forward(self,
                logits,
                probabilities,
                images,
                labels=None,
                get_logits=True,
                get_loss=False):
        """Returns the output of a forward pass for segmentation refinement."""
        logits = self.refinement_net(logits, probabilities, images)
        result = {}
        if get_logits:
            result['logits'] = logits
        if get_loss:
            result['loss'] = self.criterion(logits, labels)
        return result
