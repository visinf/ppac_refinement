# Author: Anne Wannenwetsch, TU Darmstadt (anne.wannenwetsch@visinf.tu-darmstadt.de)
# Parts of this code were adapted from https://github.com/ucbdrive/hd3
import argparse
import logging
import math
import os

import cv2
import numpy as np
import torch

import data.hd3data as hd3data
import data.flowtransforms as transforms
import hd3model
import losses
from models.hd3_ops import resize_dense_vector
from models_refine.refinement_networks import PPacNet
import prob_utils
import refinement_models
from utils import flowlib, utils

LOGGER = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description='PPAC-HD3 Inference')

    # Parameters for HD3 network.
    parser.add_argument('--encoder', type=str, help='vgg or dlaup')
    parser.add_argument('--decoder', type=str, help='resnet, or hda')
    parser.add_argument('--context', action='store_true', default=False)
    parser.add_argument(
        '--model_hd3_path', type=str, help='Path of HD3 model to be evaluated')

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
        '--model_refine_path',
        type=str,
        help='Path of PPAC refinement model to be evaluated')

    # Parameters for data loading.
    parser.add_argument(
        '--data_root', type=str, help='Root directory of data to evaluate')
    parser.add_argument(
        '--data_list', type=str, help='List of data to evaluate')
    parser.add_argument(
        '--additional_flow_masks',
        action='store_true',
        default=False,
        help='Load additional, separate invalidity flow masks')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size used for evaluation')
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of workers for data loader')

    # Parameters for evaluation.
    parser.add_argument(
        '--evaluate',
        action='store_true',
        default=False,
        help='Evaluate epe and outlier rate?')
    parser.add_argument(
        '--save_folder', type=str, help='Folder to save results')
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=False,
        help='Save flow visualizations in save_folder')
    parser.add_argument(
        '--save_inputs',
        action='store_true',
        default=False,
        help='Save HD3 inputs in save_folder for training')
    parser.add_argument(
        '--save_refined',
        action='store_true',
        default=False,
        help='Save refined PPAC outputs in save_folder')
    parser.add_argument(
        '--flow_format',
        type=str,
        default='png',
        help='Flow format used for saving: png, flo')

    return parser


def get_target_size(height_image, width_image):
    """Computes resized height and width to fit HD3 size requirements."""
    height_resized = 64 * np.array(
        [[math.floor(height_image / 64),
          math.floor(height_image / 64) + 1]])
    width_resized = 64 * np.array(
        [[math.floor(width_image / 64),
          math.floor(width_image / 64) + 1]])
    ratio = np.abs(
        np.matmul(np.transpose(height_resized), 1 / width_resized) -
        height_image / width_image)
    index = np.argmin(ratio)
    return height_resized[0, index // 2], width_resized[0, index % 2]


def save_hd3_inputs(hd3_flow, probabilities, save_folder, sub_folders, names):
    """Saves HD3 optical flow and corresponding probabilities."""
    flow_hd3 = hd3_flow.data.cpu().numpy()
    flow_hd3 = np.transpose(flow_hd3, (0, 2, 3, 1))

    probabilities_hd3 = probabilities.data.cpu().numpy()
    probabilities_hd3 = np.transpose(probabilities_hd3, (0, 2, 3, 1))

    for index in range(hd3_flow.size(0)):
        current_flow = flow_hd3[index]
        current_probabilities = probabilities_hd3[index]

        input_sub_folder = os.path.join(save_folder, sub_folders[index])
        utils.check_makedirs(input_sub_folder)
        name_flow_hd3 = os.path.join(input_sub_folder, names[index] + '.npy')
        np.save(name_flow_hd3, current_flow)

        name_probabilities_hd3 = os.path.join(input_sub_folder,
                                              names[index] + '_prob.npy')
        np.save(name_probabilities_hd3, current_probabilities)


def save_visualizations(hd3_flow, refined_flow, ground_truth,
                        visualization_folder, sub_folders, names):
    """Saves visualizations of input, refined and ground truth optical flow."""
    vis_hd3 = hd3_flow.data.cpu().numpy()
    vis_hd3 = np.transpose(vis_hd3, (0, 2, 3, 1))

    vis_refined = refined_flow.data.cpu().numpy()
    vis_refined = np.transpose(vis_refined, (0, 2, 3, 1))

    if args.evaluate:
        vis_ground_truth = ground_truth.data.cpu().numpy()
        vis_ground_truth = np.transpose(vis_ground_truth, (0, 2, 3, 1))

    for index in range(hd3_flow.size(0)):
        current_hd3 = vis_hd3[index]
        current_refined = vis_refined[index]

        rad_refined = np.sqrt(current_refined[:, :, 0]**2 +
                              current_refined[:, :, 1]**2)
        rad_hd3 = np.sqrt(current_hd3[:, :, 0]**2 + current_hd3[:, :, 1]**2)
        if args.evaluate:
            current_ground_truth = vis_ground_truth[index]
            rad_ground_truth = np.sqrt(current_ground_truth[:, :, 0]**2 +
                                       current_ground_truth[:, :, 1]**2)
            maxrad = max(-1, np.max([rad_hd3, rad_refined, rad_ground_truth]))
        else:
            maxrad = max(-1, np.max([rad_hd3, rad_refined]))

        vis_sub_folder = os.path.join(visualization_folder, sub_folders[index])
        utils.check_makedirs(vis_sub_folder)
        name_vis_hd3 = os.path.join(vis_sub_folder, names[index] + '_HD3.png')
        save_flow_visualization(current_hd3, name_vis_hd3, maxrad)
        name_vis_refined = os.path.join(vis_sub_folder,
                                        names[index] + '_refined.png')
        save_flow_visualization(current_refined, name_vis_refined, maxrad)

        if args.evaluate:
            name_vis_ground_truth = os.path.join(
                vis_sub_folder, names[index] + '_ground_truth.png')
            save_flow_visualization(current_ground_truth,
                                    name_vis_ground_truth, maxrad)


def save_flow_visualization(flow, file_path, maxrad):
    """Saves visualization of optical flow field to file_path."""
    flow = flowlib.flow_to_image(flow, maxrad=maxrad)
    flow = cv2.cvtColor(flow, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, flow)


def save_refined_flow(refined_flow, refined_folder, sub_folders, names):
    """Saves refined optical flow to save_folder."""
    refined_flow = refined_flow.data.cpu().numpy()
    refined_flow = np.transpose(refined_flow, (0, 2, 3, 1))

    for index in range(refined_flow.shape[0]):
        current_flow = refined_flow[index]
        refined_sub_folder = os.path.join(refined_folder, sub_folders[index])
        utils.check_makedirs(refined_sub_folder)
        name_refined_flow = os.path.join(refined_sub_folder,
                                         names[index] + '.' + args.flow_format)
        if args.flow_format == 'png':
            mask_blob = np.ones(current_flow.shape[:2], dtype=np.uint16)
            flowlib.write_kitti_png_file(name_refined_flow, current_flow,
                                         mask_blob)
        else:
            flowlib.write_flow(current_flow, name_refined_flow)


def main():
    global args
    args = get_parser().parse_args()
    LOGGER.info(args)

    # Get input image size and save name list.
    # Each line of data_list should contain
    # image_0, image_1, (optional) ground truth, (optional) ground truth mask.
    with open(args.data_list, 'r') as file_list:
        fnames = file_list.readlines()
        assert len(
            fnames[0].strip().split(' ')
        ) == 2 + args.evaluate + args.evaluate * args.additional_flow_masks
        input_size = cv2.imread(
            os.path.join(args.data_root, fnames[0].split(' ')[0])).shape
        if args.visualize or args.save_inputs or args.save_refined:
            names = [l.strip().split(' ')[0].split('/')[-1] for l in fnames]
            sub_folders = [
                l.strip().split(' ')[0][:-len(names[i])]
                for i, l in enumerate(fnames)
            ]
            names = [l.split('.')[0] for l in names]

    # Prepare data.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    target_height, target_width = get_target_size(input_size[0], input_size[1])
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std)])
    data = hd3data.HD3Data(
        mode='flow',
        data_root=args.data_root,
        data_list=args.data_list,
        label_num=args.evaluate + args.evaluate * args.additional_flow_masks,
        transform=transform,
        out_size=True)
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    # Setup models.
    model_hd3 = hd3model.HD3Model('flow', args.encoder, args.decoder,
                                  [4, 4, 4, 4, 4], args.context).cuda()
    model_hd3 = torch.nn.DataParallel(model_hd3).cuda()
    model_hd3.eval()

    refinement_network = PPacNet(
        args.kernel_size_preprocessing, args.kernel_size_joint,
        args.conv_specification, args.shared_filters, args.depth_layers_prob,
        args.depth_layers_guidance, args.depth_layers_joint)
    model_refine = refinement_models.EpeNet(refinement_network).cuda()
    model_refine = torch.nn.DataParallel(model_refine).cuda()
    model_refine.eval()

    # Load indicated models.
    name_hd3_model = args.model_hd3_path
    if os.path.isfile(name_hd3_model):
        checkpoint = torch.load(name_hd3_model)
        model_hd3.load_state_dict(checkpoint['state_dict'])
        LOGGER.info("Loaded HD3 checkpoint '{}'".format(name_hd3_model))
    else:
        LOGGER.info("No checkpoint found at '{}'".format(name_hd3_model))

    name_refinement_model = args.model_refine_path
    if os.path.isfile(name_refinement_model):
        checkpoint = torch.load(name_refinement_model)
        model_refine.load_state_dict(checkpoint['state_dict'])
        LOGGER.info(
            "Loaded refinement checkpoint '{}'".format(name_refinement_model))
    else:
        LOGGER.info(
            "No checkpoint found at '{}'".format(name_refinement_model))

    if args.evaluate:
        epe_hd3 = utils.AverageMeter()
        outliers_hd3 = utils.AverageMeter()
        epe_refined = utils.AverageMeter()
        outliers_refined = utils.AverageMeter()

    if args.visualize:
        visualization_folder = os.path.join(args.save_folder, 'visualizations')
        utils.check_makedirs(visualization_folder)

    if args.save_inputs:
        input_folder = os.path.join(args.save_folder, 'hd3_inputs')
        utils.check_makedirs(input_folder)

    if args.save_refined:
        refined_folder = os.path.join(args.save_folder, 'refined_flow')
        utils.check_makedirs(refined_folder)

    # Start inference.
    with torch.no_grad():
        for i, (img_list, label_list, img_size) in enumerate(data_loader):
            if i % 10 == 0:
                LOGGER.info('Done with {}/{} samples'.format(
                    i, len(data_loader)))

            img_size = img_size.cpu().numpy()
            img_list = [img.to(torch.device("cuda")) for img in img_list]
            label_list = [
                label.to(torch.device("cuda")) for label in label_list
            ]

            # Resize input images.
            resized_img_list = [
                torch.nn.functional.interpolate(
                    img, (target_height, target_width),
                    mode='bilinear',
                    align_corners=True) for img in img_list
            ]

            # Get HD3 flow.
            output = model_hd3(
                img_list=resized_img_list,
                label_list=label_list,
                get_full_vect=True,
                get_full_prob=True,
                get_epe=args.evaluate)

            # Upscale flow to full resolution.
            for level, level_flow in enumerate(output['full_vect']):
                scale_factor = 1 / 2**(6 - level)
                output['full_vect'][level] = resize_dense_vector(
                    level_flow * scale_factor, img_size[0, 1], img_size[0, 0])
            hd3_flow = output['full_vect'][-1]

            # Evaluate HD3 output if required.
            if args.evaluate:
                epe_hd3.update(
                    losses.endpoint_error(hd3_flow, label_list[0]).mean().data,
                    hd3_flow.size(0))
                outliers_hd3.update(
                    losses.outlier_rate(hd3_flow, label_list[0]).mean().data,
                    hd3_flow.size(0))

            # Upscale and interpolate flow probabilities.
            probabilities = prob_utils.get_upsampled_probabilities_hd3(
                output['full_vect'], output['full_prob'])

            if args.save_inputs:
                save_hd3_inputs(
                    hd3_flow, probabilities, input_folder,
                    sub_folders[i * args.batch_size:(i + 1) * args.batch_size],
                    names[i * args.batch_size:(i + 1) * args.batch_size])
                continue

            # Refine flow with PPAC network.
            log_probabilities = prob_utils.safe_log(probabilities)
            output_refine = model_refine(
                hd3_flow,
                log_probabilities,
                img_list[0],
                label_list=label_list,
                get_loss=args.evaluate,
                get_epe=args.evaluate,
                get_outliers=args.evaluate)

            # Evaluate refined output if required
            if args.evaluate:
                epe_refined.update(output_refine['epe'].mean().data,
                                   hd3_flow.size(0))
                outliers_refined.update(output_refine['outliers'].mean().data,
                                        hd3_flow.size(0))

            # Save visualizations of optical flow if required.
            if args.visualize:
                refined_flow = output_refine['flow']
                ground_truth = None
                if args.evaluate:
                    ground_truth = label_list[0][:, :2]
                save_visualizations(
                    hd3_flow, refined_flow, ground_truth, visualization_folder,
                    sub_folders[i * args.batch_size:(i + 1) * args.batch_size],
                    names[i * args.batch_size:(i + 1) * args.batch_size])

            # Save refined optical flow if required.
            if args.save_refined:
                refined_flow = output_refine['flow']
                save_refined_flow(
                    refined_flow, refined_folder,
                    sub_folders[i * args.batch_size:(i + 1) * args.batch_size],
                    names[i * args.batch_size:(i + 1) * args.batch_size])

    if args.evaluate:
        LOGGER.info(
            'Accuracy of HD3 optical flow:      '
            'AEE={epe_hd3.avg:.4f}, Outliers={outliers_hd3.avg:.4f}'.format(
                epe_hd3=epe_hd3, outliers_hd3=outliers_hd3))
        if not args.save_inputs:
            LOGGER.info(
                'Accuracy of refined optical flow:  '
                'AEE={epe_refined.avg:.4f}, Outliers={outliers_refined.avg:.4f}'
                .format(
                    epe_refined=epe_refined,
                    outliers_refined=outliers_refined))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
