import torch


img_height, img_width = 224, 224
img_max, img_min = 1.0, 0

cnn_model_paper = ["resnet50", "vgg16", "mobilenet_v2", "inception_v3"]
vit_model_paper = ["vit_base_patch16_224", "pit_b_224", "visformer_small", "swin_tiny_patch4_window7_224"]

cnn_model_pkg = ["vgg19", "resnet18", "resnet101", "resnext50_32x4d", "densenet121", "mobilenet_v2"]
vit_model_pkg = [
    "vit_base_patch16_224",
    "pit_b_224",
    "cait_s24_224",
    "visformer_small",
    "tnt_s_patch16_224",
    "levit_256",
    "convit_base",
    "swin_tiny_patch4_window7_224",
]

tgr_vit_model_list = [
    "vit_base_patch16_224",
    "pit_b_224",
    "cait_s24_224",
    "visformer_small",
    "deit_base_distilled_patch16_224",
    "tnt_s_patch16_224",
    "levit_256",
    "convit_base",
]

generation_target_classes = [24, 99, 245, 344, 471, 555, 661, 701, 802, 919]


def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)