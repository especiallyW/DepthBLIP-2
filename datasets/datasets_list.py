import os

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils import data
from torchvision import transforms

from utils.transform_list import EnhancedCompose, RandomColor, RandomHorizontalFlip, \
    ArrayToTensorNumpy

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _is_pil_image(img):
    return isinstance(img, Image.Image)


class Transformer(object):
    def __init__(self, args):
        if args.dataset == 'KITTI':
            self.train_transform = EnhancedCompose([
                # RandomCropNumpy((args.height, args.width)),
                RandomHorizontalFlip(),
                [RandomColor(multiplier_range=(0.9, 1.1))],
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            ])
            self.test_transform = EnhancedCompose([
                # CropNumpy((args.height, args.width)),
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            ])
        elif args.dataset == 'NYU':
            self.train_transform = EnhancedCompose([
                # RandomCropNumpy((args.height, args.width)),
                RandomHorizontalFlip(),
                [RandomColor(multiplier_range=(0.8, 1.2), brightness_mult_range=(0.75, 1.25))],
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            ])
            self.test_transform = EnhancedCompose([
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            ])

    def __call__(self, images, train=True):
        if train is True:
            return self.train_transform(images)
        else:
            return self.test_transform(images)


class NYUDataset(data.Dataset):
    """NYU Depth V2 dataloader"""

    def __init__(self, args, vis_processors, train=True, val=False, return_filename=False):
        self.args = args

        # read dataset mapping relation
        self.blip2_image_processor = vis_processors["eval"]
        self.depth_scale = 1000.0
        try:
            file_path = self.args.val_file if train and val else self.args.train_file if train else self.args.test_file
            with open(file_path, 'r') as f:
                fileset = f.readlines()
            fileset = sorted(fileset)
            self.fileset = [file for file in fileset
                            if file.split()[0].rsplit('/', 1)[0] == self.args.class_name
                            or self.args.class_name == 'all']
            self.data_root_path = os.path.join(self.args.data_root_path, 'train') if train else os.path.join(
                self.args.data_root_path, 'test')
        except FileNotFoundError as e:
            print(e.__context__)

        self.train = train
        self.transform = Transformer(args)
        self.return_filename = return_filename

    def __getitem__(self, idx):
        divided_file = self.fileset[idx].split()
        # 1-Opening image files.
        # rgb: input color image, gt: sparse depth map
        # rgb: range:(0, 1),  depth range: (0, max_depth)
        class_name, filename = divided_file[0].rsplit('/', 1)
        filename = filename.rsplit('.', 1)[0]
        # 1.1-load rgb and process rgb
        rgb_img_file = ''.join([self.data_root_path, '/', divided_file[0]])
        input_rgb_img = Image.open(rgb_img_file)
        input_rgb_img_crop = input_rgb_img.crop((40 + 20, 42 + 14, 616 - 12, 474 - 2))
        input_rgb_img = np.asarray(input_rgb_img, dtype=np.int32)
        input_rgb_img_crop = transforms.Resize((224, 224))(input_rgb_img_crop)
        preprocess_rgb_img = np.asarray(input_rgb_img_crop, dtype=np.float32) / 255.0
        preprocess_rgb_img = self.transform([preprocess_rgb_img], train=self.train)[0]
        # preprocess_rgb_img = preprocess_rgb_img.permute(2, 0, 1)
        # 1.2-load gt
        if self.args.dataset == 'NYU':
            gt_file = ''.join([self.data_root_path, '/', divided_file[1]])
            input_gt_img = Image.open(gt_file)
        else:
            print('other dataset is not supported now!')
            exit()

        # 1.3-process depth map
        if _is_pil_image(input_gt_img):
            # process gt
            # input_gt_img = input_gt_img.crop((40 + 20, 42 + 14, 616 - 12, 474 - 2))
            # input_gt_img = transforms.Resize((self.args.height, self.args.width))(input_gt_img)
            input_gt_img = np.expand_dims(np.asarray(input_gt_img, dtype=np.float32) / self.depth_scale, 2)
            input_gt_img = np.clip(input_gt_img, 0, self.args.max_depth)
            preprocess_gt_img = torch.from_numpy(input_gt_img.transpose((2, 0, 1)))
        else:
            print('location: {} is not legal image!'.format(gt_file))
            exit()

        # 2-return result
        return input_rgb_img, preprocess_rgb_img, preprocess_gt_img, filename

    def __len__(self):
        return len(self.fileset)


class KITTIDataset(data.Dataset):
    """KITTI dataloader"""

    def __init__(self, args, vis_processors, train=True, val=False, return_filename=False):
        self.args = args

        # read dataset mapping relation
        self.blip2_image_processor = vis_processors["eval"]
        self.depth_scale = 256.0
        try:
            file_path = self.args.val_file if train and val else self.args.train_file if train else self.args.test_file
            with open(file_path, 'r') as f:
                fileset = f.readlines()
            fileset = sorted(fileset)
            self.fileset = [file for file in fileset
                            if file.split()[-1] != 'None']
            self.data_root_path = os.path.join(self.args.data_root_path, 'train') if train else os.path.join(
                self.args.data_root_path, 'test')
        except FileNotFoundError as e:
            print(e.__context__)

        self.train = train
        self.transform = Transformer(args)
        self.return_filename = return_filename

    def __getitem__(self, idx):
        divided_file = self.fileset[idx].split()
        # 1-Opening image files.
        # rgb: input color image, gt: sparse depth map
        # rgb: range:(0, 1),  depth range: (0, max_depth)
        _, filename = divided_file[0].rsplit('/', 1)
        filename = filename.rsplit('.', 1)[0]
        # 1.1-load rgb and process rgb
        rgb_img_file = ''.join([self.data_root_path, '/', divided_file[0]])
        input_rgb_img = Image.open(rgb_img_file)
        # using bounding
        bound_left = (input_rgb_img.width - 1216) // 2
        bound_right = bound_left + 1216
        bound_top = input_rgb_img.height - 352
        bound_bottom = bound_top + 352
        input_rgb_img_crop = input_rgb_img.crop((bound_left, bound_top, bound_right, bound_bottom))
        input_rgb_img = transforms.Compose([
            transforms.Resize((375, 1242)),
        ])(input_rgb_img)
        input_rgb_img = np.asarray(input_rgb_img, dtype=np.int32)
        input_rgb_img_crop = transforms.Resize((224, 224))(input_rgb_img_crop)
        preprocess_rgb_img = np.asarray(input_rgb_img_crop, dtype=np.float32) / 255.0
        preprocess_rgb_img = self.transform([preprocess_rgb_img], train=self.train)[0]
        # preprocess_rgb_img = preprocess_rgb_img.permute(2, 0, 1)
        # 1.2-load gt
        gt_file = ''.join([self.data_root_path, '/', divided_file[1]])
        input_gt_img = Image.open(gt_file)

        # 1.3-process depth map
        if _is_pil_image(input_gt_img):
            # process gt
            # input_gt_img = transforms.Resize((self.args.height, self.args.width))(input_gt_img)
            # input_gt_img = input_gt_img.crop((bound_left, bound_top, bound_right, bound_bottom))
            input_gt_img = transforms.Compose([
                transforms.Resize((375, 1242)),
            ])(input_gt_img)
            input_gt_img = np.expand_dims(np.asarray(input_gt_img, dtype=np.float32) / self.depth_scale, 2)
            input_gt_img = np.clip(input_gt_img, 0, self.args.max_depth)
            preprocess_gt_img = torch.from_numpy(input_gt_img.transpose((2, 0, 1)))
        else:
            print('location: {} is not legal image!'.format(gt_file))
            exit()

        # 2-return result
        return input_rgb_img, preprocess_rgb_img, preprocess_gt_img, filename

    def __len__(self):
        return len(self.fileset)


class AllDataset(data.Dataset):
    """dataloader: supporting NYU Depth V2, KITTI"""

    def __init__(self, args, vis_processors, train=True, val=False, return_filename=False):
        if args.dataset == 'NYU':
            self.dataset = NYUDataset(args, vis_processors, train=train, val=val, return_filename=return_filename)
        elif args.dataset == 'KITTI':
            self.dataset = KITTIDataset(args, vis_processors, train=train, val=val, return_filename=return_filename)
        else:
            print('other dataset is not supported now!')
            exit()

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
