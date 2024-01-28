import torch
from torch import nn
from torch.nn import functional as F


def zeroshot_classifier(args, processor, model):
    with torch.no_grad():
        text_f_list = []
        for depth in args.depth_classes:
            for obj in args.obj_classes:
                input_text = [template.format(obj, depth) for template in args.depth_templates]  # format with class
                class_embeddings = model.extract_features({
                    "image": "",
                    "text_input": [processor["eval"](input_text[0])]
                }, mode="text").text_embeds
                class_embeddings = class_embeddings[:, 0, :]  # .mean(dim=1)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                text_f_list.append(class_embedding)
        text_f_list = torch.stack(text_f_list, dim=1)
        return text_f_list.squeeze(0)


# BLIP for Monocular Depth Estimation
class DepthBLIP(nn.Module):
    def __init__(self, model, txt_processors, args, method='second'):
        super(DepthBLIP, self).__init__()
        self.args = args
        self.bins = len(self.args.bin_list)
        self.blip2_model = model
        self.device = self.blip2_model.device
        self.txt_processors = txt_processors
        self.method = method

        if method == 'first':
            # bin_tensor: (1, bins, 1, 1)
            self.bin_tensor = (torch.tensor([1.00, 2.00, 3.00]).reshape(1, 3).
                               unsqueeze(-1).unsqueeze(-1).type(torch.float32).
                               to(self.device))  # bin_tensor=
        else:
            # text_f: (768, bins)
            self.text_f = (zeroshot_classifier(args, txt_processors, self.blip2_model).
                           type(torch.float32).to(self.device))  # init text feature
            # bin_tensor: (1, bins, 1, 1)
            self.bin_tensor = (torch.tensor(self.args.bin_list).reshape(1, self.bins).
                               unsqueeze(-1).unsqueeze(-1).type(torch.float32).
                               to(self.device))
        self.dtype = self.bin_tensor.dtype

    def forward(self, input_rgb_img, i):
        """forward to predict depth info"""
        # trans type
        input_rgb_img = input_rgb_img.type(self.dtype)

        # if chose method 'first' to finish depth prediction
        if self.method == 'first':
            depth = self.blip2_model.extract_features({
                "image": input_rgb_img,
                "text_input": [self.txt_processors["eval"]("This object is close")
                               for _ in range(input_rgb_img.shape[0])]
            }, mode="multimodal").multimodal_embeds
            depth_logits = depth.sum(dim=1).reshape(depth.shape[0], -1, 16, 3)
            depth_logits = depth_logits.permute(0, 3, 1, 2)
        # if chose method 'second' to finish depth prediction
        elif self.method == 'second':
            output_depth_list = []
            for depth in self.args.depth_classes:
                for obj in self.args.obj_classes:
                    # format with class
                    input_text = [template.format(obj, depth) for template in self.args.depth_templates]
                    output_depth = self.blip2_model.extract_features({
                        "image": input_rgb_img,
                        "text_input": [self.txt_processors["eval"](input_text[0])
                                       for _ in range(input_rgb_img.shape[0])]
                    }, mode="multimodal").multimodal_embeds
                    output_depth = output_depth.mean(dim=1)
                    # output_depth = output_depth / output_depth.norm(dim=-1, keepdim=True)
                    output_depth_list.append(output_depth)
            output_depth_list = torch.stack(output_depth_list, dim=1).cuda()
            output_depth = (output_depth_list.
                            reshape(output_depth_list.shape[0], output_depth_list.shape[1], -1, 16, 3).
                            permute(0, 1, 4, 2, 3))
            depth_logits = output_depth.max(dim=2).values
            depth_logits /= 0.05
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
        output_depth = output_depth * self.bin_tensor
        output_depth = output_depth.sum(1, keepdim=True)
        return output_depth
