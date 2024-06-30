import torch
from torch import nn


class CustomLoss(nn.Module):
    """loss function"""

    def __init__(self, dataset):
        super(CustomLoss, self).__init__()
        self.dataset = dataset
        self.alpha = 0.5
        self.crop = True

    def forward(self, pred, gt):
        if self.dataset == 'NYU':
            if self.crop:
                crop_mask = gt[0] != gt[0]
                crop_mask = crop_mask[0, :, :]
                crop_mask[45:471, 46:601] = 1

            # process gt pred
            pred_uncropped = torch.zeros_like(gt, dtype=torch.float16).cuda()
            pred_uncropped[:, :, 42 + 14:474 - 2, 40 + 20:616 - 12] = pred
            pred = pred_uncropped

            valid = (gt < 10) & (gt > 1e-3) & (pred > 1e-3)
            if self.crop:
                valid = valid & crop_mask

            valid_gt = gt[valid].clamp(1e-3, 10)
            valid_pred = pred[valid].clamp(1e-3, 10)

            # calculate result
            rmse_loss = torch.sqrt(torch.mean((valid_gt - valid_pred) ** 2))
            abs_diff_loss = torch.mean(torch.abs(valid_gt - valid_pred))
            total_loss = self.alpha * abs_diff_loss + (1 - self.alpha) * rmse_loss
            return {
                'total_loss': total_loss,
                'abs_diff_loss': abs_diff_loss,
                'rmse_loss': rmse_loss
            }

        elif self.dataset == 'KITTI':
            h, w = gt.shape[2], gt.shape[3]
            if self.crop:
                crop_mask = gt != gt
                y1, y2 = int(0.40810811 * h), int(0.99189189 * h)
                x1, x2 = int(0.03594771 * w), int(0.96405229 * w)  ### Crop used by Garg ECCV 2016
                crop_mask[:, 0, y1:y2, x1:x2] = 1
            # process gt pred
            pred_uncropped = torch.zeros_like(gt, dtype=torch.float16).cuda()
            pred_uncropped[:, :, 23:375, 13:1229] = pred
            pred = pred_uncropped

            valid = (gt < 80.0) & (gt > 1e-3)
            if self.crop:
                valid = valid & crop_mask

            valid_gt = gt[valid].clamp(1e-3, 80.0)
            valid_pred = pred[valid].clamp(1e-3, 80.0)

            # calculate result
            rmse_loss = torch.sqrt(torch.mean((valid_gt - valid_pred) ** 2))
            abs_diff_loss = torch.mean(torch.abs(valid_gt - valid_pred))
            total_loss = self.alpha * abs_diff_loss + (1 - self.alpha) * rmse_loss
            return {
                'total_loss': total_loss,
                'abs_diff_loss': abs_diff_loss,
                'rmse_loss': rmse_loss
            }
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
