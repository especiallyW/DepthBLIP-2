# -*- coding: utf-8 -*-
import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import tqdm
from torch.nn import functional as F

from blip2_extractor import load_model_and_preprocess
from dataset.datasets_list import AllDataset
from models.model import DepthBLIP
from utils.calculate_error import compute_errors_KITTI, compute_errors_NYU
from utils.logger import AverageMeter
from utils.loss import CustomLoss

parser = argparse.ArgumentParser(description='Transformer-based Monocular Depth Estimation with Attention Supervision',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Dataloader setting
parser.add_argument('--class_name', type=str, default='all', help='test data class')
parser.add_argument('--batch_size', default=32, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')

# Dataset setting
parser.add_argument('--dataset', type=str, default="NYU",
                    help="NYU / KITTI")  # Directory setting: need change directory if change dataset
parser.add_argument('--data_root_path', type=str, default="./dataset/NYU/",
                    help='data root path')
parser.add_argument('--train_file', type=str, default="./dataset/NYU/nyudepthv2_train_files_with_gt_dense.txt",
                    help='split dataset mapping file, train mode')
parser.add_argument('--val_file', type=str, default="./dataset/NYU/nyudepthv2_val_files_with_gt_dense.txt",
                    help='split dataset mapping file, evaluate mode')
parser.add_argument('--test_file', type=str,
                    default="./dataset/NYU/nyudepthv2_test_files_with_gt_dense.txt",
                    help='split dataset mapping file, test mode')
# # if change dataset, need change height and width
# # NYU=(416, 544), KITTI=(352, 1216)
parser.add_argument('--height', type=int, default=416)
parser.add_argument('--width', type=int, default=544)

# Model setting
parser.add_argument('--max_depth', default=80.0, type=float, metavar='MaxVal', help='max value of depth')
parser.add_argument('--method', default='second', type=str, help=''
                                                                 'first: BLIP-2, '
                                                                 'second: InstructBLIP')
parser.add_argument('--model_load_path', default='./checkpoints/second_NYU_bins_False/train/0/model_epoch_50.pth',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# Model setting if training
# parser.add_argument('--auto_prompt', default=False, type=bool, help='auto prompt')
parser.add_argument('--train', action='store_true', help='train or test')
parser.add_argument('--auto_bins', default=False, type=bool, help='auto bins')
parser.add_argument('--model_save_path', default='./checkpoints/', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate if train')
parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')

# Logging setting
parser.add_argument('--print_freq', default=1, type=int, metavar='N', help='print frequency')
parser.add_argument('--train_log_save', action='store_true', help='will save log record')
parser.add_argument('--test_log_save', action='store_true', help='will save log record')
parser.add_argument('--log_result_dir', type=str, default="./log_results/",
                    help='need save path if save logger result when train')

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
                    # default=[4.00, 7.00, 8.00, 8.50, 9.25, 9.75, 10.00],
                    default=[1.00, 1.75, 2.25, 2.50, 2.75, 3.00, 3.50],
                    help='depth bin list')


def create_new_run_dir(prepare_save_path, args):
    """build new dir for save model checkpoint / train log / test log"""
    save_path = os.path.join(prepare_save_path,
                             ''.join([args.method, '_', args.dataset, '_bins_', 'True' if args.auto_bins else 'False']),
                             'train' if args.train is True else 'test')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, str(len(os.listdir(save_path))))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path


def save_test_result(args, error_string):
    # save log result if setting save log record
    if args.test_log_save:
        with open(os.path.join(args.log_save_path, 'test_result.txt'), 'a') as f:
            f.write(f'\n\n---->method: {args.method}')
            f.write(f'\n\n---->dataset: {args.dataset}')
            f.write(f'\n\n---->class_name: {args.class_name}')
            f.write(f'\n---->depth_templates: {args.depth_templates}')
            f.write(f'\n---->obj_classes: {args.obj_classes}')
            f.write(f'\n---->depth_classes: {args.depth_classes}')
            f.write(f'\n---->bin_list: {args.bin_list}')
            f.write('\n* * Avg {}'.format(error_string))


def save_train_result(args, train_error_dict, val_error_dict, epoch):
    # save log result if setting save log record
    if args.train_log_save:
        with open(os.path.join(args.log_save_path, 'train_result.txt'), 'a') as f:
            f.write(f"\n\n------------epoch {epoch}------------")
            f.write('\ntraining: {}'.format(', '.join([f'{k}: {v:.3f}' for k, v in train_error_dict.items()])))
            f.write('\nvalid: {}'.format(', '.join([f'{k}: {v:.3f}' for k, v in val_error_dict.items()])))


def train(model, train_loader, valid_loader, optimizer, criterion, epoch, args):
    """training model for each epoch"""
    # one epoch, loading dataset and training
    model.train()
    errors = AverageMeter(i=2)
    pbar = tqdm.tqdm(train_loader, desc=f"Training: epoch {epoch + 1}", ncols=120, colour='green')
    for i, (input_rgb_img, preprocess_rgb_img, preprocess_gt_img, filename) in enumerate(
            pbar):  # enumerate(train_loader):
        # rgb_data: (batch_size, channel_num, H, W)
        # gt_data: (batch_size, 1, H, W)
        preprocess_rgb_img = preprocess_rgb_img.cuda()
        preprocess_gt_img = preprocess_gt_img.cuda()
        # flip rgb img
        preprocess_rgb_img_flip = torch.flip(preprocess_rgb_img, [3])
        output_depth = model(preprocess_rgb_img, i)
        output_depth_flip = model(preprocess_rgb_img_flip, i)
        # weighted sum
        # output_depth: (batch_size, 1, H, W)
        output_depth = 0.5 * (output_depth + torch.flip(output_depth_flip, [3]))
        output_depth = F.interpolate(output_depth, size=[args.height, args.width],
                                     mode='bilinear', align_corners=True)
        # calculate loss
        losses = criterion(output_depth, preprocess_gt_img)
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()
        # print loss
        errors.update([losses['abs_diff_loss'].item(), losses['rmse_loss'].item()])
        train_error_dict = {'Abs diff loss': errors.avg[0], 'RMSE loss': errors.avg[1]}
        train_error_dict['lr'] = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_error_dict)
    # validate model for each epoch
    val_error_dict = validate(model, valid_loader, args, epoch)
    # save test result
    save_train_result(args, train_error_dict, val_error_dict, epoch)


def validate(model, val_loader, args, epoch):
    """validating model for each epoch"""
    # global device
    if args.dataset == 'KITTI':
        error_names = ['abs_diff', 'sq_rel', 'a1', 'a2', 'a3', 'abs_rel', 'rmse', 'rmse_log']
    elif args.dataset == 'NYU':
        error_names = ['abs_diff', 'a1', 'a2', 'a3', 'abs_rel', 'log10', 'rmse']
    elif args.dataset == 'Make3D':
        error_names = ['abs_diff', 'abs_rel', 'ave_log10', 'rmse']

    # one epoch, loading dataset and validating
    model.eval()
    errors = AverageMeter(i=len(error_names))
    pbar = tqdm.tqdm(val_loader, desc=f"Validating: epoch {epoch + 1}", ncols=180, colour='yellow')
    for i, (input_rgb_img, preprocess_rgb_img, preprocess_gt_img, filename) in enumerate(pbar):
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
        if args.dataset == 'KITTI':
            err_result = compute_errors_KITTI(preprocess_gt_img, output_depth, crop=True)
        elif args.dataset == 'NYU':
            err_result = compute_errors_NYU(preprocess_gt_img, output_depth, crop=True)
        errors.update(err_result)
        # print error
        val_error_dict = {name: error for name, error in
                          zip(error_names[0:len(error_names)], errors.avg[0:len(errors.avg)])}
        pbar.set_postfix(val_error_dict)
    return val_error_dict


def testing(model, val_loader, args):
    """testing model for eval mode"""
    # global device
    if args.dataset == 'KITTI':
        error_names = ['abs_diff', 'sq_rel', 'a1', 'a2', 'a3', 'abs_rel', 'rmse', 'rmse_log']
    elif args.dataset == 'NYU':
        error_names = ['abs_diff', 'a1', 'a2', 'a3', 'abs_rel', 'log10', 'rmse']
    elif args.dataset == 'Make3D':
        error_names = ['abs_diff', 'abs_rel', 'ave_log10', 'rmse']

    # loading dataset and testing
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
        if args.dataset == 'KITTI':
            err_result = compute_errors_KITTI(preprocess_gt_img, output_depth, crop=True)
        elif args.dataset == 'NYU':
            err_result = compute_errors_NYU(preprocess_gt_img, output_depth, crop=True)
        errors.update(err_result)
        # print frequency
        if i % args.print_freq == 0:
            error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in
                                     zip(error_names[0:len(error_names)], errors.val[0:len(errors.avg)]))
            print('valid: {}/{} {}'.format(i, length, error_string))

    # final metric result
    error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in
                             zip(error_names[0:len(error_names)], errors.avg[0:len(errors.avg)]))
    print(' * * Avg {}'.format(error_string))
    save_test_result(args, error_string)


def main():
    args = parser.parse_args()
    print("==> No Distributed Training")
    print('==> Index of using GPU: ', args.gpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    torch.manual_seed(args.seed)

    ######################  Pretraining model loading part  ##########################
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # model type choice: blip-2/InstructBLIP
    name = 'depth_blip2_opt' if args.method == 'first' else 'depth_blip2_vicuna_instruct'
    model_type = 'pretrain_opt2.7b' if args.method == 'first' else 'vicuna7b'
    os.environ['TORCH_HOME'] = './config/blip2-opt-2.7b' if args.method == 'first' else './config/blip2-vicuna-instruct-7b'
    blip2_model, vis_processors, txt_processors = load_model_and_preprocess(name=name,
                                                                            model_type=model_type,
                                                                            is_eval=True,
                                                                            device=device)

    ######################  Data loading part  ##########################
    if args.dataset == 'KITTI':
        args.temperature = 0.1 if args.method == 'first' else 0.8
        args.max_depth = 80.0
    elif args.dataset == 'NYU':
        args.temperature = 2.4 if args.method == 'first' else 0.5
        args.max_depth = 10.0
    print("==> Mode: ", "Training" if args.train == True else "Testing")
    print("==> Method: ", args.method)
    print("==> Auto prompt: ", args.auto_prompt)
    print("==> Auto bins: ", args.auto_bins)
    print("==> Dataset: ", args.dataset)
    print("==> Data height: {}, width: {} ".format(args.height, args.width))

    # if using train mode
    if args.train:
        # loading train set and val set
        train_set = AllDataset(args, vis_processors, train=True, val=False)
        valid_set = AllDataset(args, vis_processors, train=True, val=True)
        print('==> train samples num: {}  '.format(len(train_set)))
        print('==> valid samples num: {}  '.format(len(valid_set)))
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            pin_memory=True, drop_last=False,
            num_workers=args.workers, sampler=None)
        valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=args.batch_size, shuffle=False,
            pin_memory=True, drop_last=False,
            num_workers=args.workers, sampler=None)
        # if is fine-tune Q-Former, load test set after training finish
        if not args.auto_bins:
            test_set = AllDataset(args, vis_processors, train=False)
            test_loader = torch.utils.data.DataLoader(
                test_set, batch_size=args.batch_size, shuffle=False,
                pin_memory=True, drop_last=False,
                num_workers=args.workers, sampler=None)
            print('==> test samples num: {}  '.format(len(test_set)))
            cudnn.benchmark = True
    else:  # if only use test mode
        test_set = AllDataset(args, vis_processors, train=False)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False,
            pin_memory=True, drop_last=False,
            num_workers=args.workers, sampler=None)
        print('==> test samples num: {}  '.format(len(test_set)))
        cudnn.benchmark = True

    print('==> Data class name: {}  '.format(args.class_name))
    print(
        f"==> Language prompts: {args.depth_templates}, \n"
        f"==> Obj class: {args.obj_classes}, \n"
        f"==> Semantic distance categories: {args.depth_classes},"
        f"==> Values of Depth bins: {args.bin_list if not args.auto_bins else 'Auto'} \n"
    )

    ######################  setting Network  ###################
    print("==> creating model")
    model = DepthBLIP(blip2_model, txt_processors, args, flag=2)
    model = torch.nn.DataParallel(model)

    # modify model grad info
    for name, params in model.named_parameters():
        if args.auto_bins:
            if not name.startswith('module.mlp'):
                params.requires_grad = False
    print("===============================================")
    print("Total trainable parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print("Total parameters: {}".format(sum(p.numel() for p in model.parameters())))
    print("===============================================")

    ######################  eval dataset  ###################
    # define optimizer, loss function and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = CustomLoss(args.dataset)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # training model or testing
    args.log_save_path = create_new_run_dir(args.log_result_dir, args)
    args.model_save_path = create_new_run_dir(args.model_save_path, args)
    if args.train:
        for epoch in range(args.epochs):
            train(model, train_loader, valid_loader, optimizer, criterion, epoch, args)
            scheduler.step()
            if args.auto_bins:
                torch.save(model.module.mlp.state_dict(),
                           os.path.join(args.model_save_path, f'model_epoch_{epoch + 1}.pth'))
            else:
                if epoch == args.epochs - 1:
                    torch.save(model.module.blip2_model.state_dict(),
                               os.path.join(args.model_save_path, f'model_epoch_{epoch + 1}.pth'))
                    testing(model, test_loader, args)
    else:
        if args.auto_bins:
            model.module.mlp.load_state_dict(torch.load(args.model_load_path))
        else:
            model.module.blip2_model.load_state_dict(torch.load(args.model_load_path))
        testing(model, test_loader, args)
        print(args.dataset, "validation finish")


if __name__ == "__main__":
    main()
