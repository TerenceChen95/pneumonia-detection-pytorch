# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-

import re
import os
import numpy as np
import torch
from torch import nn
from torchvision import models
import argparse
from skimage import io
import cv2
from ref.grad_cam import GradCAM
from ref.guided_back_propagation import GuidedBackPropagation
import pydicom
from PIL import Image

def get_net(net_name, weight_path=None):
    pretrain = weight_path is None  
    if net_name in ['densenet', 'densenet121']:
        net = models.densenet121(pretrained=pretrain)
    else:
        raise ValueError('invalid network name:{}'.format(net_name))
    if weight_path is not None and net_name.startswith('densenet'):
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(weight_path)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        net.load_state_dict(state_dict, strict=False)
    elif weight_path is not None:
        net.load_state_dict(torch.load(weight_path), strict=False)
    return net


def get_last_conv_name(net):
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def prepare_input(image):
    image = image.copy()

    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # channel first
    image = image[np.newaxis, ...]  # 增加batch维

    return torch.tensor(image, requires_grad=True)


def gen_cam(image, mask):
    # transfer mask to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return norm_image(cam), heatmap


def norm_image(image):
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_gb(grad):
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image_dicts, input_image_name, network, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)


def main(args):
    img_arr = pydicom.read_file(args.image_path).pixel_array
    img = Image.fromarray(img_arr).convert('RGB')
    img = np.array(img)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    inputs = prepare_input(img)
    # output image
    image_dict = {}
    # get the model
    net = get_net(args.network, args.weight_path)
    # Grad-CAM
    layer_name = get_last_conv_name(net) if args.layer_name is None else args.layer_name
    grad_cam = GradCAM(net, layer_name)
    mask = grad_cam(inputs, args.class_id)  # cam mask
    image_dict['cam'], image_dict['heatmap'] = gen_cam(img, mask)
    grad_cam.remove_handlers()
    # GuidedBackPropagation
    gbp = GuidedBackPropagation(net)
    inputs.grad.zero_()  
    grad = gbp(inputs)

    gb = gen_gb(grad)
    image_dict['gb'] = gb
    # generate Guided Grad-CAM
    cam_gb = gb * mask[..., np.newaxis]
    image_dict['cam_gb'] = norm_image(cam_gb)

    save_image(image_dict, os.path.basename(args.image_path), args.network, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='densenet121',
                        help='ImageNet classification network')
    parser.add_argument('--image-path', type=str, default='/home/tianshu/pneumonia/dataset/stage_2_test_images/c1f55e7e-4065-4dc0-993e-a7c1704c6036.dcm',
                        help='input image path')
    parser.add_argument('--weight-path', type=str, default='./checkpoint.pth',
                        help='weight path of the model')
    parser.add_argument('--layer-name', type=str, default=None,
                        help='last convolutional layer name')
    parser.add_argument('--class-id', type=int, default=None,
                        help='class id')
    parser.add_argument('--output-dir', type=str, default='result',
                        help='output directory to save results')
    arguments = parser.parse_args()

    main(arguments)
