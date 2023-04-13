import numpy as np

import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50, fcn_resnet101
import requests
from PIL import Image
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg') # Necessary to run matplotlib
from tqdm import tqdm

from pytorch_grad_cam.utils.image import show_cam_on_image

import skimage

from collections import OrderedDict

class TorchSegmentationWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)['out']

def vis_predict(image, model, preprocess_transform, DEVICE = 'cpu', mask = None, box = None, fig_name = None, vis = True):
    if preprocess_transform is None:
        input_tensor = image.clone()
        image = image.permute(1, 2, 0).numpy()
    else:
        input_tensor = preprocess_transform(image)
    
    
    if mask is not None:
        input_tensor = input_tensor * mask
    output = model(input_tensor.unsqueeze(0).to(DEVICE))[0].detach().cpu()
    output = output.argmax(axis = 0)

    if box is None:
        box = (0, 0, 0, 0)
    rect_image = image.copy()
    rect_image = cv2.rectangle(rect_image, (box[2], box[0]), (box[3], box[1]), 255, 10)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.imshow(output)
    plt.title('To Explain')
    plt.subplot(1, 3, 3)
    plt.imshow(rect_image)
    if fig_name is not None:
        plt.savefig(fig_name, bbox_inches = 'tight')
    if vis is True:
        plt.show()
    plt.close()
    return image, output, rect_image

def dice(a, b):
    return 2*(a & b).sum()/(a.sum() + b.sum())

def generate_masks(n_masks, input_size, p1 = 0.1, initial_mask_size = (7, 7), binary = True):
    # cell size in the upsampled mask
    Ch = np.ceil(input_size[0] / initial_mask_size[0])
    Cw = np.ceil(input_size[1] / initial_mask_size[1])

    resize_h = int((initial_mask_size[0] + 1) * Ch)
    resize_w = int((initial_mask_size[1] + 1) * Cw)

    masks = []

    for _ in range(n_masks):
        # generate binary mask
        binary_mask = torch.randn(
            1, 1, initial_mask_size[0], initial_mask_size[1])
        binary_mask = (binary_mask < p1).float()

        # upsampling mask
        if binary:
            mask = F.interpolate(
                binary_mask, (resize_h, resize_w), mode='nearest')#, align_corners=False)
        else:
            mask = F.interpolate(
                binary_mask, (resize_h, resize_w), mode='bilinear', align_corners=False)

        # random cropping
        i = np.random.randint(0, Ch)
        j = np.random.randint(0, Cw)
        mask = mask[:, :, i:i+input_size[0], j:j+input_size[1]]

        masks.append(mask)

    masks = torch.cat(masks, dim=0)   # (N_masks, 1, H, W)

    return masks

def rise_segmentation(masks, image, model, preprocess_transform, target = None, n_masks = None, box = None, DEVICE = 'cpu', vis = True, vis_skip = 1):
    if preprocess_transform is None:
        input_tensor = image.clone()
        image = image.permute(1, 2, 0).numpy()
    else:
        input_tensor = preprocess_transform(image)

    
    if box is None:
        y_start, y_end, x_start, x_end = 0, image.shape[0], 0, image.shape[1]
    else:
        y_start, y_end, x_start, x_end = box[0], box[1], box[2], box[3]

    coef = []

    if n_masks is None:
        n_masks = len(masks)

    output = model(input_tensor.unsqueeze(0).to(DEVICE))[0].detach().cpu()
    output_1 = output.argmax(axis = 0)
    output_a = output_1[y_start:y_end, x_start:x_end]
    
    if target is None:
        target = output_a.max().item()
    
    for index, mask in tqdm(enumerate(masks[:n_masks])):
        #input_tensor = preprocess_transform(image)
        #output = model(input_tensor.unsqueeze(0).to(DEVICE))[0].detach().cpu()
        #output_1 = output.argmax(axis = 0)

        input_tensor_1 = input_tensor * mask
        output = model(input_tensor_1.unsqueeze(0).to(DEVICE))[0].detach().cpu()
        output_2 = output.argmax(axis = 0)
        
        #output_a = output_1[y_start:y_end, x_start:x_end]
#         if target is None:
#             target = output_a.max().item()
        output_b = output_2[y_start:y_end, x_start:x_end]
    
        DICE = dice(output_a == target, output_b == target)
        coef.append(DICE)

        if vis == True:
            if index % vis_skip == 0:
                plt.figure()
                plt.subplot(1, 3, 1)
                plt.imshow(output_1)
                plt.subplot(1, 3, 2)
                plt.imshow(output_2)
                plt.subplot(1, 3, 3)
                plt.imshow(mask[0])
                plt.show()
    return coef

def rise_aggregated(image, masks, coef, fig_name = None, vis = True):
    aggregated_mask = np.zeros(masks[0][0].shape)

    for i, j in zip(masks[:len(coef)], coef):
        aggregated_mask += i[0].detach().cpu().numpy() * j.item()

    max_, min_ = aggregated_mask.max(), aggregated_mask.min() 
    aggregated_mask = np.uint8(255 * (aggregated_mask - min_) / (max_ - min_))
    overlaid = show_cam_on_image(image/255, aggregated_mask/255, use_rgb=True)

    title = 'RISE'

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.imshow(aggregated_mask)
    plt.title(title)
    plt.subplot(1, 3, 3)
    plt.imshow(overlaid)
    if fig_name is not None:
            plt.savefig(fig_name, bbox_inches = 'tight')
    if vis is True:
        plt.show()
    plt.close()        

    return aggregated_mask, overlaid

def save_grad(x, gradients):
     x.register_hook(lambda z: gradients.append(z))

def seg_grad_cam(image, model, preprocess_transform, target = None, target_layer = None, box = None, DEVICE = 'cpu', method_index = 0, fig_base_name = None, fig_name = None, vis_base = True, vis = True, negative_gradient = False, pool_size = None, pool_mode = np.max, reshape_transformer = False):
    if preprocess_transform is None:
        input_tensor = image.clone()
        image = image.permute(1, 2, 0).numpy()
        max_, min_ = image.max(), image.min()
        image = np.uint8(255 * (image - min_) / (max_ - min_))

    else:
        input_tensor = preprocess_transform(image)

    activations, gradients = [], []

    handle_1 = target_layer.register_forward_hook(lambda x, y, z: activations.append(z))
    handle_2 = target_layer.register_forward_hook(lambda x, y, z: save_grad(z, gradients))

    model.zero_grad()
    output = model(input_tensor.unsqueeze(0).to(DEVICE))

    if box is None:
        y_start, y_end, x_start, x_end = 0, image.shape[0], 0, image.shape[1]
    else:
        y_start, y_end, x_start, x_end = box[0], box[1], box[2], box[3]

    if target is None:
        target = output[0].argmax(0).max().item()

    mask = output[0].argmax(0).detach().cpu().numpy()
    mask_uint8 = 255 * np.uint8(mask == target)
    mask_float = np.float32(mask == target)
    mask_mask = np.zeros(mask_float.shape)
    mask_mask[y_start:y_end, x_start:x_end] = 1
    mask_float = mask_float * mask_mask
    mask_uint8 = np.uint8(mask_uint8 * mask_mask)

    if negative_gradient == True:
        loss = -(output[0, target, :, :] * torch.tensor(mask_float).to(DEVICE)).sum()
    else:
        loss = (output[0, target, :, :] * torch.tensor(mask_float).to(DEVICE)).sum()
    loss.backward()

    activations = activations[0][0].detach().cpu().numpy()
    gradients = gradients[0][0].detach().cpu().numpy()
    
    if reshape_transformer == True:
        activations = np.reshape(activations, (14, 14, activations.shape[1]))
        gradients = np.reshape(gradients, (14, 14, gradients.shape[1]))
        activations = np.transpose(activations, (2, 0, 1))
        gradients = np.transpose(gradients, (2, 0, 1))
    
    if method_index == 0:
        coef = gradients.sum(axis = (1, 2))
        coef = coef[:, None, None]
        grayscale_cam = coef * activations
        grayscale_cam = grayscale_cam.sum(axis = 0)
    elif method_index == 1:
        if pool_size is not None:
            pooled = skimage.measure.block_reduce(gradients, (1, pool_size, pool_size), pool_mode)
            pooled = np.transpose(pooled, (1, 2, 0))
            #gradients = cv2.resize(pooled, (gradients.shape[1], gradients.shape[0]), interpolation = cv2.INTER_NEAREST)
            gradients = skimage.transform.resize(pooled, (gradients.shape[1], gradients.shape[2]), order = 0)
            gradients = np.transpose(gradients, (2, 0, 1))
            
        grayscale_cam = gradients * activations
        grayscale_cam = grayscale_cam.sum(axis = 0)
        
    grayscale_cam = np.maximum(grayscale_cam, 0)
    grayscale_cam = cv2.resize(grayscale_cam, (image.shape[1], image.shape[0]))
    max_, min_ = grayscale_cam.max(), grayscale_cam.min() 
    grayscale_cam = np.uint8(255 * (grayscale_cam - min_) / (max_ - min_))
    grayscale_cam = grayscale_cam / 255.0
    
    overlaid = show_cam_on_image(image/255, grayscale_cam, use_rgb=True)
    
    title = 'Seg-Grad-CAM' if method_index == 0 else 'Seg-HiResCAM'
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(np.repeat(mask_uint8[:, :, None], 3, axis=-1))

    if fig_base_name is not None:
        plt.savefig(fig_name, bbox_inches = 'tight')
    if vis_base is True:
        plt.show()
    plt.close()

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.imshow(grayscale_cam)
    plt.title(title)
    plt.subplot(1, 3, 3)
    plt.imshow(overlaid)
    
    if fig_name is not None:
        plt.savefig(fig_name, bbox_inches = 'tight')
    if vis is True:
        plt.show()
    plt.close()   
                    
    handle_1.remove()
    handle_2.remove()

    return grayscale_cam, overlaid



def utility_dilation(image, model, preprocess_transform, target = None, box = None, DEVICE = 'cpu', vis = True):
  if preprocess_transform is None:
          input_tensor = image.clone()
          image = image.permute(1, 2, 0).numpy()
          max_, min_ = image.max(), image.min()
          image = np.uint8(255 * (image - min_) / (max_ - min_))
  else:
      input_tensor = preprocess_transform(image)

  output = model(input_tensor.unsqueeze(0).to(DEVICE))

  if box is None:
      y_start, y_end, x_start, x_end = 0, image.shape[0], 0, image.shape[1]
  else:
      y_start, y_end, x_start, x_end = box[0], box[1], box[2], box[3]

  if target is None:
      target = output[0].argmax(0).max().item()

  mask = output[0].argmax(0).detach().cpu().numpy()
  mask_uint8 = 255 * np.uint8(mask == target)
  mask_float = np.float32(mask == target)
  mask_mask = np.zeros(mask_float.shape)
  mask_mask[y_start:y_end, x_start:x_end] = 1
  mask_float = mask_float * mask_mask
  mask_uint8 = np.uint8(mask_uint8 * mask_mask)

  plt.figure()
  plt.subplot(1, 2, 1)
  plt.imshow(image)
  plt.subplot(1, 2, 2)
  plt.imshow(np.repeat(mask_uint8[:, :, None], 3, axis=-1))
  if vis is True:
    plt.show()
  plt.close()
    
  return image, mask_float

def dc(result, reference, label):
    result = result == label
    reference = reference == label
    #result = np.atleast_1d(result.astype(np.bool))
    #reference = np.atleast_1d(reference.astype(np.bool))
    intersection = np.count_nonzero(result & reference)
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)
    dc = 2. * intersection / float(size_i1 + size_i2)
    return dc

def dilation(image, model, preprocess_transform, target = None, box = None, DEVICE = 'cpu',
             mask = None, kernel_size = 5, threshold = 0.2, iterations = 100, original_prediction = None, skip_vis = 10):
    dil_mask = mask.copy()  
    dil_mask[dil_mask < threshold] = 0
    dil_mask[dil_mask > threshold] = 1
    images, masks = [], []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    for i in range(0, iterations):
        dil_mask = cv2.dilate(dil_mask.astype(np.uint8), kernel, iterations = i)
        if image.shape[0] <= 3:
          im = image * dil_mask[None, :, :]
        else:
          im = image * dil_mask[:, :, None]
        if i % skip_vis == 0:
            vis = True
        else:
            vis = False
        im_iter, mask_iter = utility_dilation(image = im, model = model, preprocess_transform = 
                                              preprocess_transform, target = target, box = box, 
                                              DEVICE = DEVICE, vis = vis)
        images.append(im_iter)
        masks.append(mask_iter)

    
    if box is None:
        if image.shape[0] <= 3:
            y_start, y_end, x_start, x_end = 0, image.shape[1], 0, image.shape[2]
        else:
            y_start, y_end, x_start, x_end = 0, image.shape[0], 0, image.shape[1]
    else:
        y_start, y_end, x_start, x_end = box[0], box[1], box[2], box[3]

    mask_float = np.float32(original_prediction == target)
    mask_mask = np.zeros(mask_float.shape)
    print(y_start, y_end, x_start, x_end)
    mask_mask[y_start:y_end, x_start:x_end] = 1
    mask_float = mask_float * mask_mask

    a = []
    for i in masks:

        print(dc(mask_float, i, 1))
        a.append(dc(mask_float, i, 1))
    
    plt.figure()
    plt.scatter(range(len(a)), a)
    
    return images, masks, a
