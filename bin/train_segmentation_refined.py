# Author: Anne Wannenwetsch, TU Darmstadt (anne.wannenwetsch@visinf.tu-darmstadt.de)
# Parts of this code were adapted from https://github.com/ucbdrive/hd3
import argparse
import logging
import os
import shutil

import torch
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image

from datasets import datasets_segmentation, data_utils
import losses
from models_refine.refinement_networks import PPacNet
import prob_utils
import refinement_models
from utils import utils

LOGGER = logging.getLogger(__name__)


# Setup
def get_parser():
    parser = argparse.ArgumentParser(description='PPAC Segmentation Training')

    # Parameters for PPAC refinement network.
    parser.add_argument(
        '--depth_layers_guidance',
        nargs="+",
        type=int,
        default=[],
        help='Depth of guidance layers in PPAC refinement network')
    parser.add_argument(
        '--depth_layers_prob',
        nargs="+",
        type=int,
        default=[],
        help='Depth of probability layers in PPAC refinement network')
    parser.add_argument(
        '--kernel_size_preprocessing',
        type=int,
        default=5,
        help='Kernel size of guidance and probability branches')
    parser.add_argument(
        '--conv_specification',
        nargs="+",
        type=str,
        default=[],
        help='Type of joint layers in PPAC refinement network')
    parser.add_argument(
        '--depth_layers_joint',
        nargs="+",
        type=int,
        default=[],
        help='Depth of joint layers in PPAC refinement network')
    parser.add_argument(
        '--shared_filters',
        action='store_true',
        default=False,
        help='Use shared filters in combination branch?')
    parser.add_argument(
        '--kernel_size_joint',
        type=int,
        default=7,
        help='Kernel size in combination branch')
    parser.add_argument(
        '--pretrained_model_refine',
        type=str,
        help='Path to pretrained PPAC refinement model')

    # Parameters for data loading.
    parser.add_argument(
        '--dataset_name', type=str, help='Name of train dataset')
    parser.add_argument(
        '--data_root',
        type=str,
        help='Root directory of train/validation data')
    parser.add_argument(
        '--logits_root',
        type=str,
        help='Root directory of saved logits input data')
    parser.add_argument('--train_list', type=str, help='List of train data')
    parser.add_argument('--val_list', type=str, help='List of validation data')
    parser.add_argument(
        '--num_segmentation_classes',
        type=int,
        help='Number of segmentation classes')
    parser.add_argument(
        '--invalid_label',
        type=float,
        default=None,
        help='Value used for invalid labels')
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of workers for data loader')

    # Parameters for learning
    parser.add_argument(
        '--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size used for training')
    parser.add_argument(
        '--base_lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument(
        '--preprocessing_lr',
        type=float,
        default=None,
        help='Learning rate used for guidance/probability branch')
    parser.add_argument(
        '--fixed_gaussian_weights',
        action='store_true',
        default=False,
        help='Use fixed Gaussian weights in combination branch, '
             'e.g. to determine pre-processing learning rate?')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='Weight decay used in training')

    # Parameters for validation during training
    parser.add_argument(
        '--batch_size_val',
        type=int,
        default=1,
        help='Batch size for validation during training')
    parser.add_argument(
        '--evaluation-frequency',
        type=int,
        default=1,
        help='Evaluate every x epochs')
    parser.add_argument(
        '--evaluate_only',
        action='store_true',
        default=False,
        help='Perform only a single evaluation cycle, no training?')

    # Parameters for outputs
    parser.add_argument(
        '--save_step', type=int, default=50, help='Save model every x epochs')
    parser.add_argument(
        '--save_folder',
        type=str,
        default='model',
        help='Folder to save model and training summary')
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=False,
        help='Save segmentation maps in save_folder')
    return parser


def main():
    global args, writer
    args = get_parser().parse_args()
    writer = SummaryWriter(args.save_folder)
    LOGGER.info(args)

    refinement_network = PPacNet(
        args.kernel_size_preprocessing,
        args.kernel_size_joint,
        args.conv_specification,
        args.shared_filters,
        args.depth_layers_prob,
        args.depth_layers_guidance,
        args.depth_layers_joint,
        fixed_gaussian_weights=args.fixed_gaussian_weights,
        bias_zero_init=True)
    model_refine = refinement_models.CrossEntropyNet(refinement_network).cuda()
    model_refine = torch.nn.DataParallel(model_refine).cuda()
    LOGGER.info('Used PPAC refinement model:')
    LOGGER.info(model_refine)

    if args.pretrained_model_refine:
        name_refinement_model = args.pretrained_model_refine
        if os.path.isfile(name_refinement_model):
            checkpoint = torch.load(name_refinement_model)
            model_refine.load_state_dict(checkpoint['state_dict'])
            LOGGER.info("Loaded pretrained PPAC checkpoint '{}'".format(
                name_refinement_model))
        else:
            LOGGER.info(
                "No checkpoint found at '{}'".format(name_refinement_model))

    # Prepare data.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    crop_shape = get_crop_shape(args.dataset_name)
    train_transform = datasets_segmentation.AugmenterSegmentation(
        mean, std, crop_shape)
    val_transform = datasets_segmentation.AugmenterSegmentation(mean, std)
    train_data = datasets_segmentation.SegmentationDataset(
        data_root=args.data_root,
        data_list=args.train_list,
        logits_root=args.logits_root,
        invalid_label=args.invalid_label,
        transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)
    val_data = datasets_segmentation.SegmentationDataset(
        data_root=args.data_root,
        data_list=args.val_list,
        logits_root=args.logits_root,
        invalid_label=args.invalid_label,
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    if args.evaluate_only:
        miou_meter_input = losses.MiouMetric(args.num_segmentation_classes)
        miou_meter = losses.MiouMetric(args.num_segmentation_classes)
        loss_val, miou_val, loss_val_input, miou_val_input = validate(
            val_loader, model_refine, miou_meter, miou_meter_input)
        LOGGER.info(
            'Performance of input semantic segmentation:   cross entropy={:.4f}, mIoU={:.4f}'
                .format(loss_val_input, miou_val_input))
        LOGGER.info(
            'Performance of refined semantic segmentation: cross entropy={:.4f}, mIoU={:.4f}'
                .format(loss_val, miou_val))
        return

    # Prepare learning.
    if args.preprocessing_lr:
        optimizer = torch.optim.Adam(
            [{
                'params':
                    model_refine.module.refinement_net.layers_joint.parameters()
            },
                {
                    'params':
                        model_refine.module.refinement_net.network_guidance.
                            parameters(),
                    'lr':
                        args.preprocessing_lr
                },
                {
                    'params':
                        model_refine.module.refinement_net.network_prob.parameters(),
                    'lr':
                        args.preprocessing_lr
                }],
            lr=args.base_lr,
            weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(
            model_refine.parameters(),
            lr=args.base_lr,
            weight_decay=args.weight_decay)
    scheduler = get_lr_scheduler(optimizer, args.dataset_name)

    best_miou = 0
    miou_meter = losses.MiouMetric(args.num_segmentation_classes)

    # Start learning.
    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        loss_train = train(train_loader, model_refine, optimizer, epoch,
                           args.batch_size)
        writer.add_scalar('loss_train', loss_train, epoch)

        is_best = False
        if epoch % args.evaluation_frequency == 0:
            torch.cuda.empty_cache()
            miou_meter.reset()
            loss_val, miou_val = validate(val_loader, model_refine, miou_meter)
            LOGGER.info('Epoch {}. Validation cross entropy: {:.4f}, '
                        'Validation mIoU: {:.4f}'.format(
                epoch, loss_val, miou_val))
            writer.add_scalar('loss_val', loss_val, epoch)
            writer.add_scalar('miou_val', miou_val, epoch)
            is_best = miou_val > best_miou
            best_miou = max(miou_val, best_miou)

        filename = os.path.join(args.save_folder, 'model_refine_latest.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model_refine.cpu().state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_miou': best_miou,
        }, filename)
        model_refine.cuda()
        if is_best:
            shutil.copyfile(
                filename,
                os.path.join(args.save_folder, 'model_refine_best.pth'))
        if epoch % args.save_step == 0:
            shutil.copyfile(
                filename, args.save_folder + '/train_refine_epoch_' +
                          str(epoch) + '.pth')


def train(train_loader, model_refine, optimizer, epoch, batch_size):
    """Performs one training pass."""
    loss_meter = utils.AverageMeter()
    model_refine.train()

    for i, (image, input_logits, label) in enumerate(train_loader):
        if image.size(0) < batch_size:
            continue

        image = image.to(torch.device("cuda"))
        input_logits = input_logits.to(torch.device("cuda"))
        label = label.to(torch.device("cuda"))

        probabilities = torch.nn.functional.softmax(input_logits, dim=1)
        probabilities = prob_utils.safe_log(probabilities)
        output_refine = model_refine(
            input_logits, probabilities, image, labels=label, get_loss=True)
        total_loss = output_refine['loss'].sum()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss_meter.update(total_loss.mean().data, image.size(0))
        current_iter = (epoch - 1) * len(train_loader) + i + 1
        writer.add_scalar('loss_train_batch',
                          loss_meter.val.cpu().numpy(), current_iter)

    return loss_meter.avg.cpu().numpy()


def validate(val_loader, model_refine, miou_meter, miou_meter_input=None):
    """Performs one validation pass."""
    loss_meter = utils.AverageMeter()
    model_refine.eval()

    if miou_meter_input is not None:
        cross_entropy_loss = losses.CrossEntropySegmentationCalculator()
        loss_meter_input = utils.AverageMeter()

    if args.visualize:
        with open(args.val_list, 'r') as file_list:
            fnames = file_list.readlines()
            names = [l.strip().split(' ')[0].split('/')[-1] for l in fnames]
            sub_folders = [
                l.strip().split(' ')[0][:-len(names[i])]
                for i, l in enumerate(fnames)
            ]
            names = [l.split('.')[0] for l in names]
        visualization_folder = os.path.join(args.save_folder, 'visualizations')
        utils.check_makedirs(visualization_folder)
        colormap = data_utils.create_colormap(args.dataset_name)

    with torch.no_grad():
        for i, (image, input_logits, label) in enumerate(val_loader):
            image = image.to(torch.device("cuda"))
            input_logits = input_logits.to(torch.device("cuda"))
            label = label.to(torch.device("cuda"))

            probabilities = torch.nn.functional.softmax(input_logits, dim=1)
            probabilities = prob_utils.safe_log(probabilities)
            output_refine = model_refine(
                input_logits,
                probabilities,
                image,
                labels=label,
                get_loss=True,
                get_logits=True)
            logits = output_refine['logits']

            miou_meter.update(logits, label)
            loss_meter.update(output_refine['loss'].mean().data, image.size(0))
            if miou_meter_input is not None:
                miou_meter_input.update(input_logits, label)
                loss_meter_input.update(
                    cross_entropy_loss(input_logits, label).mean().data,
                    image.size(0))

            if args.visualize:
                save_visualizations(
                    image, input_logits, logits, label, visualization_folder,
                    sub_folders[i * args.batch_size_val:(i + 1) *
                                                        args.batch_size_val],
                    names[i * args.batch_size_val:(i + 1) *
                                                  args.batch_size_val], colormap)

    if miou_meter_input is not None:
        return loss_meter.avg.cpu().numpy(), miou_meter.get(
        ), loss_meter_input.avg.cpu().numpy(), miou_meter_input.get()
    return loss_meter.avg.cpu().numpy(), miou_meter.get()


def get_crop_shape(dataset_name):
    """Returns cropping shape corresponding to dataset_name."""
    if dataset_name == 'Pascal':
        return (200, 272)
    else:
        raise ValueError('Unknown dataset name {}'.format(dataset_name))


def get_lr_scheduler(optimizer, dataset_name=None):
    if dataset_name == 'Pascal':
        milestones = [501]
    else:
        raise ValueError('Unknown dataset name {}'.format(dataset_name))
    LOGGER.info('Milestones: {}'.format(str(milestones)))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5)
    return scheduler


def save_visualizations(image, input_logits, logits, label,
                        visualization_folder, sub_folders, names, colormap):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    input_segmentation = torch.argmax(input_logits, dim=1)
    input_segmentation = input_segmentation.int().data.cpu().numpy()

    refined_segmentation = torch.argmax(logits, dim=1)
    refined_segmentation = refined_segmentation.int().data.cpu().numpy()

    ground_truth = label[:, 0].int().data.cpu().numpy()

    for index in range(input_logits.size(0)):
        current_image = image[index]
        current_input_seg = input_segmentation[index]
        current_refined_seg = refined_segmentation[index]
        current_ground_truth = ground_truth[index]

        # Invert normalization of input image
        for j, (channel_mean, channel_std) in enumerate(zip(mean, std)):
            current_image[j] = current_image[j] * channel_std + channel_mean
        current_image = np.transpose(current_image.cpu().numpy(),
                                     (1, 2, 0)) * 255.

        vis_sub_folder = os.path.join(visualization_folder, sub_folders[index])
        utils.check_makedirs(vis_sub_folder)
        name_vis_refined = os.path.join(vis_sub_folder,
                                        names[index] + '_refined.png')
        save_segmentation_visualization(current_refined_seg, current_image,
                                        colormap, name_vis_refined)

        name_vis_refined = os.path.join(vis_sub_folder,
                                        names[index] + '_input.png')
        save_segmentation_visualization(current_input_seg, current_image,
                                        colormap, name_vis_refined)

        name_vis_ground_truth = os.path.join(
            vis_sub_folder, names[index] + '_ground_truth.png')
        save_segmentation_visualization(current_ground_truth, current_image,
                                        colormap, name_vis_ground_truth)


def save_segmentation_visualization(segmentation, image, colormap, save_path):
    segmentation = colormap[segmentation]
    visualization = image * 0.5 + segmentation * 0.5
    visualization = Image.fromarray(visualization.astype('uint8'))
    visualization.save(save_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
