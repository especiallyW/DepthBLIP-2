import os


def extract_summary(kitti_dir, save_path):
    # extract projection for kitti eigen split-testing
    line_list = []
    first_dir = os.listdir(kitti_dir)
    for i in first_dir:
        second_dir = os.listdir(os.path.join(kitti_dir, i))
        for j in second_dir:
            third_dir = os.listdir(os.path.join(kitti_dir, i, j))
            for k in third_dir:
                if k.startswith('image'):
                    fourth_dir = os.listdir(os.path.join(kitti_dir, i, j, k))
                    for l in fourth_dir:
                        all_file = os.listdir(os.path.join(kitti_dir, i, j, k, l))
                        for file in all_file:
                            rgb_img_path = os.path.join(i, j, k, l, file).replace('\\', '/')
                            gt_img_path = os.path.join(i, j, 'proj_depth', 'groundtruth', k, file).replace('\\', '/')
                            if not os.path.exists(os.path.join(kitti_dir, gt_img_path)):
                                gt_img_path = 'None'
                            line_string = ' '.join([rgb_img_path, gt_img_path]) + '\n'
                            line_list.append(line_string)
    # got projection file
    with open(save_path + 'eigen_test_files_with_gt_dense.txt', 'w') as f:
        f.writelines(line_list)


if __name__ == '__main__':
    kitti_dir = 'KITTI/'
    save_path = 'KITTI/'
    extract_summary(kitti_dir, save_path)
