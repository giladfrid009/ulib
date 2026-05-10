import torch


img_height, img_width = 224, 224
img_max, img_min = 1.0, 0

generation_target_classes = [24, 99, 245, 344, 471, 555, 661, 701, 802, 919]


def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)
