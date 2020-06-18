# Probabilistic Pixel-Adaptive Refinement Networks

This source code release accompanies the paper

[**Probabilistic Pixel-Adaptive Refinement Networks**](http://openaccess.thecvf.com/content_CVPR_2020/html/Wannenwetsch_Probabilistic_Pixel-Adaptive_Refinement_Networks_CVPR_2020_paper.html) \
Anne S. Wannenwetsch, Stefan Roth.
In CVPR 2020.

The code in this repository allows to refine outputs of (probabilistic)
deep networks with image-adaptive, confidence-aware
convolutions. Applications to the tasks of optical flow and semantic
segmentation refinement are illustrated.

Contact: Anne Wannenwetsch (anne.wannenwetsch@visinf.tu-darmstadt.de)

Requirements
------------
The code was tested with Python 3.6, PyTorch 1.0.0 and Cuda 9.0.

Further requirements can be installed with
```bash
pip install -r requirements.txt
```

We further require the code that accompanies the paper

*Pixel-Adaptive Convolutional Neural Networks.
Hang Su, Varun Jampani, Deqing Sun, Orazio Gallo, Erik Learned-Miller, and Jan Kautz.
CVPR 2019*

which underlies our probabilistic pixel-adaptive convolutions (PPACs).
Please download the corresponding repository from
<https://github.com/NVlabs/pacnet>, e.g. using
```bash
git clone https://github.com/NVlabs/pacnet.git
```

For the application to optical flow, please also download the code of
the paper

*Hierarchical Discrete Distribution Decomposition for Match Density Estimation.
Zhichao Yin, Trevor Darrell, Fisher Yu.
CVPR 2019*

from <https://github.com/ucbdrive/hd3>, e.g.
```bash
git clone https://github.com/ucbdrive/hd3.git
```

PPAC refinement requires to apply some small changes to the
original HD3 code. Moreover, we adjust the code to allow invalidity
masks also for the Sintel dataset. Please apply the
provided patch to the downloaded HD3 repository. For instance, you
could do the following:
```bash
cp ~/ppac_refinement/0001-Apply-PPAC-changes.patch ~/hd3
cd ~/hd3
git am -3 < 0001-Apply-PPAC-changes.patch
```
Please adapt the paths accordingly if the directories `ppac_refinement`
and `hd3` are not located in your home folder.

Before running the code, make sure to set `PYTHONPATH` appropriately,
e.g. by performing the following:
```bash
cd ~/ppac_refinement
export PYTHONPATH=$PYTHONPATH:`pwd`/src
export PYTHONPATH=$PYTHONPATH:~/pacnet
export PYTHONPATH=$PYTHONPATH:~/hd3
```
Again, please adapt the paths if the directories are not located
in your home folder.

Training and inference procedure
--------------------------------
We provide code for the refinement of two different estimate types:
optical flow fields and semantic segmentation maps.
Depending on the specified options (see section below), the different 
networks are built and trained or evaluated.

Sample scripts to illustrate the usage of the training
functions `bin/train_flow_refined.py` as well as
`bin/train_segmentation_refined.py` and especially the default settings
of the available parameters can be found in the directory `scripts`.
Please note that we trained all our networks on a setup using two GPUs
simultaneously.

Moreover, we include a function `bin/inference_hd3_refined.py` which
allows to directly estimate (and evaluate) PPAC-HD3 optical flow
given input image pairs and a pre-trained HD3 and PPAC refinement
checkpoint.


Test
----

For testing purposes, we have included sample images and ground truth
from the Pascal VOC 2012, Sintel and KITTI datasets in
`sample_data/images` as well as `sample_data/flow`,
`sample_data/invalid` and `sample_data/segmentation`, respectively.
To test PPAC refinement on these samples, please run
```bash
bash scripts/train_refine_sintel.sh
bash scripts/train_refine_kitti.sh
bash scripts/train_refine_pascal.sh
```
using option `--evaluate_only` and specifying an appropriate save folder
with `--save_folder`.
One should expect AEE=0.33 for Sintel, AEE=0.55 for the KITTI sample and
mIoU=0.98 on Pascal as performance of the refined estimates.


Baseline estimates as input to training procedure
-------------------------------------------------

The training procedure requires as input the saved
estimates of the underlying, task-specific neural networks. Estimates
have to be provided in a directory specified by the parameter
`--flow_root` or `--logits_root` and should have `.npy` format.
For our optical flow experiments, we used different (fine-tuned) HD3
models as described in section 6.1 of our paper which can be downloaded
from <https://github.com/ucbdrive/hd3>.
For semantic segmentation on Pascal VOC 2012, we applied the checkpoint
`xception_coco_voc_trainaug` of DeepLabV3+ which can be found at
<https://github.com/qixuxiang/deeplabv3plus/blob/master/g3doc/model_zoo.md>.
Sample files for both tasks can be found in `/sample_data/inputs_flow` and
`/sample_data/inputs_segmentation`.
Please note that PPAC refinement takes network predictions at full
resolution as inputs, i.e. you should save the estimates
_after_ the (bilinear) upsampling step.

For optical flow, one can save the required HD3 flow fields and
probabilities by calling `inference_hd3_refined.py` with option
`--save_inputs`.
While we created our Sintel and KITTI benchmark uploads with this
method, please note that the underlying flow fields for Tables 1 and 2
of our paper were saved using the function `train.py` of the original
HD3 repository. This method applies a different approach to rescale
input images and thus leads to (slightly) different HD3 results.


Important parameters for PPAC refinement networks
-------------------------------------------------
* `dataset_name`: Name of dataset used for training, e.g.
determines learning rate schedule
* `data_root`: Root directory of training/validation data
* `flow/logits_root`: Folder with input flow/segmentation
* `train/val_list`: List of samples used for training and validation,
respectively
* `base_lr`: Learning rate used for all parameters without explicitly
defined learning rate
* `preprocessing_lr`: Learning rate used for guidance and probability
branch if specified (otherwise `base_lr` is used)
* `batch_size(_val)`: Batch size used during training/validation
* `epochs`: Total number of training epochs
* `save_folder`: Folder to which summaries, visualizations, refined
estimates etc are saved
* `kernel_size_preprocessing/joint`: Kernel size used in preprocessing
and combination branch, respectively
* `depth_layers_guidance/prob/joint`: List of number of channels used in
guidance, probability and combination branch, respectively
* `conv_specification`: Determines type of convolutions used in the
combination branch (p=PPACs, c=standard convolutions)
* `shared_filters`: Determines if the convolution weight is shared
across all channels of the input estimates
* `pretrained_model_refine/model_refine_path`: Path to pretrained PPAC
refinement model
* `evaluate_only`: Perform only one validation path of the provided
training procedure
* `visualize`: Save visualizations of inputs, refined estimates and if
available ground truth
* `save_inputs`: Save HD3 inputs as required during training
* `save_refined`: Save refined estimates

Please note that not all of the above options are applicable to all
train/inference methods. To see all available parameters and the
corresponding explanations, you can use
```bash
python <train_or_inference_method.py> --help
```
replacing `<...>` with the respective training or inference function.


Data splits
-----------

The data splits used in this paper for training, validation and test are
the same as in our previous paper *Learning Task-Specific Generalized
Convolutions in the Permutohedral Lattice* and can be found
in the corresponding repository at
<https://github.com/visinf/semantic_lattice/tree/master/experiments/lists>.

Pretrained networks
-------------------

In the directory `checkpoints`, we provide pre-trained PPAC networks for
optical flow and semantic segmentation which underlie the results
presented in Tables 3, 4, and 5 of the main paper.
Please refer to the paper as well as the supplemental material for the
specifics of these networks.

Advanced normalization PAC network
----------------------------------
For illustration purposes, we finally include a non-probabilistic PAC
network (PacNetAdvancedNormalization) using our advanced normalization
scheme in `src/models_refine/refinement_network.py`. This network is
applicable to data without probabilities. Please note that we
concatenated image guidance data and probabilities for the PAC
experiments in our paper.

Citation
--------

If you use our code, please cite our CVPR 2020 paper:

    @inproceedings{Wannenwetsch:2020:PPA,
        title = {Probabilistic Pixel-Adaptive Refinement Networks},
        author = {Anne S. Wannenwetsch and Stefan Roth},
        booktitle = {CVPR},
        year = {2020}}

Acknowledgements
----------------
Parts of this code are inspired and/or adapted from code available in the
following repositories:
* <https://github.com/ucbdrive/hd3>
* <https://github.com/NVlabs/pacnet>
* <https://github.com/visinf/semantic_lattice>

Corresponding files are labeled accordingly.
