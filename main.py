# -*- coding: utf-8 -*-
import argparse
import os
import torch
import torch.utils.data
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from lavis.models import load_model_and_preprocess

from models.model import DepthBLIP
from utils.logger import AverageMeter
from datasets.datasets_list import NYUDataset
from utils.calculate_error import compute_each_errors

parser = argparse.ArgumentParser(description='Transformer-based Monocular Depth Estimation with Attention Supervision',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Directory setting
parser.add_argument('--test_file', type=str,
                    default="./datasets/DepthV2/nyudepthv2_test_files_with_gt_dense.txt",
                    help='split dataset mapping file')
parser.add_argument('--data_root_path', type=str, default="./datasets/NYU_Depth_V2",
                    help='data root path')

# Dataloader setting
parser.add_argument('--class_name', type=str, default='all', help='test data class')
parser.add_argument('--batch_size', default=32, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--dataset', type=str, default="NYU")
parser.add_argument('--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')

# Model setting
parser.add_argument('--height', type=int, default=416)
parser.add_argument('--width', type=int, default=544)
parser.add_argument('--max_depth', default=80.0, type=float, metavar='MaxVal', help='max value of depth')

# Logging setting
parser.add_argument('--print_freq', default=5, type=int, metavar='N', help='print frequency')
parser.add_argument('--log_save', action='store_true', help='will save log record')
parser.add_argument('--log_result_dir', type=str, default="./log_results",
                    help='need save path if save logger result')

# GPU parallel process setting
parser.add_argument('--gpu_num', type=str, default="0,1,2,3", help='force available gpu index')

# Prompt design setting
parser.add_argument('--depth_templates', type=str,
                    default=['This {} is {}'], nargs='+', help='prompt template')
parser.add_argument('--obj_classes', type=str, nargs='+', default=['object'], help='obj class')
parser.add_argument('--depth_classes', type=str, nargs='+',
                    default=['giant', 'extremely close', 'close', 'not in distance', 'a little remote', 'far',
                             'unseen'],
                    help='depth class')
parser.add_argument('--bin_list', type=float, nargs='+',
                    default=[1.00, 1.75, 2.25, 2.50, 2.75, 3.00, 3.50],
                    help='depth bin list')


def save(args, error_string):
    # save log result if setting save log record
    if args.log_save:
        if not os.path.exists(args.log_result_dir):
            os.makedirs(args.log_result_dir)
        with open(os.path.join(args.log_result_dir, 'result.txt'), 'a') as f:
            f.write(f'\n\n---->class_name: {args.class_name}')
            f.write(f'\n---->depth_templates: {args.depth_templates}')
            f.write(f'\n---->obj_classes: {args.obj_classes}')
            f.write(f'\n---->depth_classes: {args.depth_classes}')
            f.write(f'\n---->bin_list: {args.bin_list}')
            f.write('\n* * Avg {}'.format(error_string))


def validate(model, val_loader, args):
    # global device
    if args.dataset == 'KITTI':
        error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3', 'rmse', 'rmse_log']
    elif args.dataset == 'NYU':
        error_names = ['abs_diff', 'a1', 'a2', 'a3', 'abs_rel', 'log10', 'rmse']
    elif args.dataset == 'Make3D':
        error_names = ['abs_diff', 'abs_rel', 'ave_log10', 'rmse']

    model.eval()
    length = len(val_loader)
    errors = AverageMeter(i=len(error_names))
    # iter dataset
    for i, (input_rgb_img, preprocess_rgb_img, preprocess_gt_img, filename) in enumerate(val_loader):
        if preprocess_rgb_img.ndim != 4 and preprocess_rgb_img[0] == False:
            continue
        # rgb_data: (batch_size, channel_num, H, W)
        # gt_data: (batch_size, 1, H, W)
        preprocess_rgb_img = preprocess_rgb_img.cuda()
        preprocess_gt_img = preprocess_gt_img.cuda()
        # flip rgb img
        preprocess_rgb_img_flip = torch.flip(preprocess_rgb_img, [3])
        with torch.no_grad():
            output_depth = model(preprocess_rgb_img, i)
            output_depth_flip = model(preprocess_rgb_img_flip, i)
            # weighted sum
            # output_depth: (batch_size, 1, H, W)
            output_depth = 0.5 * (output_depth + torch.flip(output_depth_flip, [3]))
            output_depth = F.interpolate(output_depth, size=[args.height, args.width],
                                         mode='bilinear', align_corners=True)
        # calculate error
        err_result = compute_each_errors(preprocess_gt_img, output_depth, crop=True)
        errors.update(err_result)
        # print frequency
        if i % args.print_freq == 0:
            error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in
                                     zip(error_names[0:len(error_names)], errors.avg[0:len(errors.avg)]))
            print('valid: {}/{} {}'.format(i, length, error_string))

    # final metric result
    error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in
                             zip(error_names[0:len(error_names)], errors.avg[0:len(errors.avg)]))
    print(' * * Avg {}'.format(error_string))
    save(args, error_string)


def main():
    args = parser.parse_args()
    print("==> No Distributed Training")
    print('==> Index of using GPU: ', args.gpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    torch.manual_seed(args.seed)

    ######################  Data loading part  ##########################
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    blip2_model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor",
                                                                            model_type="pretrain_vitL",
                                                                            is_eval=True,
                                                                            device=device)

    test_set = NYUDataset(args, vis_processors, train=False)
    print("==> Dataset: ", args.dataset)
    print("==> Data height: {}, width: {} ".format(args.height, args.width))
    print('==> test samples num: {}  '.format(len(test_set)))
    print('==> Data class name: {}  '.format(args.class_name))
    print(
        f"==> Language prompts: {args.depth_templates}, \n"
        f"==> Obj class: {args.obj_classes}, \n"
        f"==> Semantic distance categories: {args.depth_classes}, \n"
        f"==> Values of Depth bins: {args.bin_list} \n"
    )

    test_sampler = None
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, sampler=test_sampler)
    cudnn.benchmark = True

    if args.dataset == 'KITTI':
        args.max_depth = 80.0
    elif args.dataset == 'NYU':
        args.temperature = 0.1
        args.max_depth = 10.0

    ######################  setting Network  ###################
    print("==> creating model")
    if args.dataset == 'KITTI':
        model = DepthBLIP(blip2_model, txt_processors, args)
    if args.dataset == 'NYU':
        model = DepthBLIP(blip2_model, txt_processors, args)

    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    model = torch.nn.DataParallel(model)
    print("===============================================")
    print("Total parameters: {}".format(num_params))
    print("===============================================")

    ######################  eval dataset  ###################
    validate(model, test_loader, args)
    print(args.dataset, "validation finish")


if __name__ == "__main__":
    main()
