import torch
from torch import nn
from torch.nn import functional as F

from clip import clip


def zeroshot_classifier(args, model):
    with torch.no_grad():
        text_f_list = []
        for depth in args.depth_classes:
            for obj in args.obj_classes:
                input_text = [template.format(obj, depth) for template in args.depth_templates]  # format with class
                texts = clip.tokenize(input_text).cuda()  # tokenize
                class_embeddings = model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                text_f_list.append(class_embedding)
        text_f_list = torch.stack(text_f_list, dim=1)
    return text_f_list.squeeze(0)


# BLIP for Monocular Depth Estimation
class DepthBLIP(nn.Module):
    def __init__(self, model, txt_processors, args, flag=1):
        super(DepthBLIP, self).__init__()
        self.args = args
        self.bins = len(self.args.bin_list)
        self.blip2_model = model
        self.device = self.blip2_model.device
        self.txt_processors = txt_processors
        self.method = args.method

        if self.args.auto_bins:
            if flag == 1:
                self.mlp = nn.Sequential(
                    nn.Linear(self.bins, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(64, self.bins),
                    nn.Sigmoid()
                )
            elif flag == 2:
                self.mlp = nn.Sequential(
                    nn.Linear(self.bins, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, self.bins),
                    nn.Sigmoid()
                )
            else:
                self.mlp = nn.Sequential(
                    nn.Linear(self.bins, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, self.bins),
                    nn.Sigmoid()
                )

        if self.method == 'first':
            # bin_tensor: (1, bins, 1, 1)
            self.bin_tensor = (torch.tensor(self.args.bin_list).reshape(1, self.bins).type(torch.float32).
                               to(self.device))  # bin_tensor
            # text_f: (768, bins)
            self.text_f = (zeroshot_classifier(args, clip.load('RN50')[0]).
                           type(torch.float32).to(self.device))  # init text feature
        else:
            # bin_tensor: (1, bins, 1, 1)
            self.bin_tensor = (torch.tensor(self.args.bin_list).reshape(1, self.bins).type(torch.float32).
                               to(self.device))

        self.bin_min = min(self.args.bin_list)
        self.bin_max = max(self.args.bin_list)
        self.dtype = self.bin_tensor.dtype

    def forward(self, input_rgb_img, i):
        """forward to predict depth info"""
        # trans type
        input_rgb_img = input_rgb_img.type(self.dtype)
        if self.args.auto_bins:
            bin_tensor = self.bin_min + (self.mlp(self.bin_tensor) ** 2) * (self.bin_max - self.bin_min)
        else:
            bin_tensor = self.bin_tensor

        # if chose method 'first' to finish depth prediction
        if self.method == 'first':
            img_f = self.blip2_model.extract_features({
                "image": input_rgb_img,
            })
            text_features = self.text_f

            # @: dot product of two vectors
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            depth_logits = 100. * img_f.type(self.dtype) @ text_features
            depth_logits = depth_logits.permute(0, 2, 1).reshape(-1, self.bins, 4, 8)
            depth_logits /= self.args.temperature
        # if chose method 'second' to finish depth prediction
        elif self.method == 'second':
            output_depth_list = []
            # feature fusion by text and image
            for depth in self.args.depth_classes:
                for obj in self.args.obj_classes:
                    # format with class
                    input_text = [template.format(obj, depth) for template in self.args.depth_templates]
                    output_depth = self.blip2_model.extract_features({
                        "image": input_rgb_img,
                        "text_input": [self.txt_processors["eval"](input_text[0])
                                       for _ in range(input_rgb_img.shape[0])]
                    })
                    output_depth = output_depth.mean(dim=1)
                    # output_depth = output_depth / output_depth.norm(dim=-1, keepdim=True)
                    output_depth_list.append(output_depth)
                # feature dimension processing
                output_depth_list = torch.stack(output_depth_list, dim=1).cuda()
                if self.args.dataset == 'KITTI':
                    dim = (output_depth_list.shape[0], output_depth_list.shape[1], 1, 16, 48)
                else:
                    dim = (output_depth_list.shape[0], output_depth_list.shape[1], 3, 16, -1)
            output_depth = output_depth_list.reshape(*dim)
            depth_logits = output_depth.max(dim=2).values
            depth_logits /= self.args.temperature
        # if chose method 'third' to finish depth prediction
        else:
            img_f = self.blip2_model.extract_features({
                "image": input_rgb_img,
                "text_input": [self.txt_processors["eval"]("")]
            }, mode="image").image_embeds
            # @: dot product of two vectors
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            depth_logits = 100. * img_f.type(self.dtype) @ self.text_f
            depth_logits = depth_logits.permute(0, 2, 1).reshape(-1, self.bins, 4, 8)
            depth_logits /= self.args.temperature

        # depth information
        # output_depth: (batch_size, 1, H, W)
        output_depth = F.softmax(depth_logits, dim=1)
        output_depth = output_depth * bin_tensor.view(1, -1, 1, 1)
        output_depth = output_depth.sum(1, keepdim=True)
        return output_depth
