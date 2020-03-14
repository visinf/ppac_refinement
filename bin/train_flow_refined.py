# Author: Anne Wannenwetsch, TU Darmstadt (anne.wannenwetsch@visinf.tu-darmstadt.de)
# Parts of this code were adapted from https://github.com/ucbdrive/hd3
import argparse
import logging
import os
import shutil

import torch
from tensorboardX import SummaryWriter

from datasets import datasets_flow
from models_refine.refinement_networks import PPacNet
import prob_utils
import refinement_models
from utils import utils

LOGGER = logging.getLogger(__name__)


# Setup
def get_parser():
    parser = argparse.ArgumentParser(description='PPAC Flow Training')

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
        '--flow_root',
        type=str,
        help='Root directory of saved flow input data')
    parser.add_argument('--train_list', type=str, help='List of train data')
    parser.add_argument('--val_list', type=str, help='List of validation data')
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
        default=8,
        help='Batch size used for training')
    parser.add_argument(
        '--base_lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument(
        '--preprocessing_lr',
        type=float,
        default=None,
        help='Learning rate used for guidance/probability branch')
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

    return parser


def main():
    global args, writer
    args = get_parser().parse_args()
    writer = SummaryWriter(args.save_folder)
    LOGGER.info(args)

    refinement_network = PPacNet(
        args.kernel_size_preprocessing, args.kernel_size_joint,
        args.conv_specification, args.shared_filters, args.depth_layers_prob,
        args.depth_layers_guidance, args.depth_layers_joint)
    model_refine = refinement_models.EpeNet(refinement_network).cuda()
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
    train_transform = datasets_flow.AugmenterFlow(mean, std, crop_shape)
    val_transform = datasets_flow.AugmenterFlow(mean, std)
    train_data = datasets_flow.FlowDataset(
        data_root=args.data_root,
        data_list=args.train_list,
        flow_root=args.flow_root,
        transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)
    val_data = datasets_flow.FlowDataset(
        data_root=args.data_root,
        data_list=args.val_list,
        flow_root=args.flow_root,
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    if args.evaluate_only:
        epe_val, outliers_val = validate(val_loader, model_refine)
        LOGGER.info(
            'Validation EPE: {:.4f}, Validation outliers: {:.4f}'.format(
                epe_val, outliers_val))
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

    best_epe = 1e9

    # Start learning.
    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        loss_train = train(train_loader, model_refine, optimizer, epoch,
                           args.batch_size)
        writer.add_scalar('loss_train', loss_train, epoch)

        is_best = False
        if epoch % args.evaluation_frequency == 0:
            torch.cuda.empty_cache()
            epe_val, outliers_val = validate(val_loader, model_refine)
            LOGGER.info(
                'Epoch {}. Validation EPE: {:.4f}, Validation outliers: {:.4f}'
                .format(epoch, epe_val, outliers_val))
            writer.add_scalar('epe_val', epe_val, epoch)
            writer.add_scalar('outliers_val', outliers_val, epoch)
            is_best = epe_val < best_epe
            best_epe = min(epe_val, best_epe)

        filename = os.path.join(args.save_folder, 'model_refine_latest.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model_refine.cpu().state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_epe': best_epe
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

    for i, (image, input_flow, probabilities,
            label) in enumerate(train_loader):
        if image.size(0) < batch_size:
            continue

        image = image.to(torch.device("cuda"))
        input_flow = input_flow.to(torch.device("cuda"))
        probabilities = probabilities.to(torch.device("cuda"))
        label = label.to(torch.device("cuda"))

        probabilities = prob_utils.safe_log(probabilities)
        output_refine = model_refine(
            input_flow,
            probabilities,
            image,
            label_list=[label],
            get_loss=True)
        total_loss = output_refine['loss'].sum()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss_meter.update(total_loss.mean().data, image.size(0))
        current_iter = (epoch - 1) * len(train_loader) + i + 1
        writer.add_scalar('loss_train_batch',
                          loss_meter.val.cpu().numpy(), current_iter)

    return loss_meter.avg.cpu().numpy()


def validate(val_loader, model_refine):
    """Performs one validation pass."""
    epe_meter = utils.AverageMeter()
    outlier_meter = utils.AverageMeter()
    model_refine.eval()

    with torch.no_grad():
        for (image, input_flow, probabilities, label) in val_loader:
            image = image.to(torch.device("cuda"))
            input_flow = input_flow.to(torch.device("cuda"))
            probabilities = probabilities.to(torch.device("cuda"))
            label = label.to(torch.device("cuda"))

            probabilities = prob_utils.safe_log(probabilities)
            output_refine = model_refine(
                input_flow,
                probabilities,
                image,
                label_list=[label],
                get_loss=False,
                get_epe=True,
                get_outliers=True)

            epe_meter.update(output_refine['epe'].mean().data, image.size(0))
            outlier_meter.update(output_refine['outliers'].mean().data,
                                 image.size(0))

    return epe_meter.avg, outlier_meter.avg


def get_crop_shape(dataset_name):
    """Returns cropping shape corresponding to dataset_name."""
    if 'MPISintel' in dataset_name:
        return (384, 768)
    elif 'KITTI' in dataset_name:
        return (320, 896)
    else:
        raise ValueError('Unknown dataset name {}'.format(dataset_name))


def get_lr_scheduler(optimizer, dataset_name=None):
    """Returns learning rate scheduler corresponding to dataset_name."""
    if dataset_name == 'KITTI':
        milestones = [100, 200, 300, 400]
    elif dataset_name == 'KITTI_full':
        milestones = [80, 160, 240, 320]
    elif dataset_name == 'MPISintel':
        milestones = [100, 200, 300, 400]
    elif dataset_name == 'MPISintel_full':
        milestones = [82, 164, 246, 328]
    else:
        raise ValueError('Unknown dataset name {}'.format(dataset_name))
    LOGGER.info('Milestones: {}'.format(str(milestones)))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5)
    return scheduler


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
