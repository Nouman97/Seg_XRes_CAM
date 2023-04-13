# Seg-XRes-CAM

Code for Seg-XRes-CAM

## Instructions - For google colab:

!git clone https://github.com/Nouman97/Seg_XRes_CAM.git

!pip3 install grad-cam

```python

import os

os.chdir('Seg_XRes_CAM')

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from seg_xres_cam import TorchSegmentationWrapper
from visualize import visualize_algos
from seg_xres_cam import dilation
import requests


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

preprocess_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True).to(DEVICE)
model = TorchSegmentationWrapper(model)
model = model.eval()
image_url = "http://images.cocodataset.org/val2017/000000544811.jpg"
image = np.array(Image.open(requests.get(image_url, stream=True).raw).convert("RGB"))

method_dict = {'Seg-Grad-CAM': 0, 'Seg-XRes-CAM': 1}

n_masks, p1, window_size = 2000, 0.1, (7, 7)
input_size = image.shape
target_layer = model.model.backbone.layer4
method_indexes = [method_dict['Seg-Grad-CAM'], method_dict['Seg-XRes-CAM'], method_dict['Seg-XRes-CAM']]
pool_sizes, pool_modes, reshape_transformer = [0, 1, 2], [None, np.mean, np.mean], False
fig_size = (30, 50)
vis, vis_base, vis_rise, grid = False, False, False, True
preprocess_transform = preprocess_transform

target = 3
y_start, y_end, x_start, x_end = 150, 320, 450, 630

box = (y_start, y_end, x_start, x_end)
results_1_natural, results_masks_1 = visualize_algos(image, model, preprocess_transform = preprocess_transform,
                    target = target, target_layer = target_layer, box = box, 
                    DEVICE = DEVICE, method_indexes = method_indexes, 
                    fig_base_name = None, fig_name = None, vis_base = vis_base, 
                    vis = vis, negative_gradient = False, pool_sizes = pool_sizes, 
                    pool_modes = pool_modes, reshape_transformer = reshape_transformer, 
                    n_masks = n_masks, input_size = input_size, p1 = p1, 
                    initial_mask_size = window_size, image_vis = image, vis_rise = vis_rise, 
                    fig_size = fig_size, grid = grid)

ims_med, masks_med, dice_med = dilation(image, model, preprocess_transform, target = target, box = box, DEVICE = DEVICE,
          mask = results_masks_1[-2], kernel_size = 5, threshold = 0.2, iterations = 10, original_prediction = results_1_natural[1],
          skip_vis = 2)
```  

## Saliency Results

![image](https://user-images.githubusercontent.com/127871419/226063304-87789063-5ea6-412b-83cc-ac11b95a02f9.png)

## Dice progression with dilation starting with Seg-XRes-CAM's saliency map

![image](https://user-images.githubusercontent.com/127871419/226063330-7ce1d45f-7e70-44ee-a28a-8d8d7fd7b647.png)

![image](https://user-images.githubusercontent.com/127871419/226063430-3a784bdf-f9c8-4c47-90f1-746c7fedc0bb.png)

![image](https://user-images.githubusercontent.com/127871419/226063469-f19fdb5d-d38e-4ed6-a327-34f8db9c405c.png)

![image](https://user-images.githubusercontent.com/127871419/226063477-5d25cb03-ce16-402a-a7df-80d144204b85.png)

![image](https://user-images.githubusercontent.com/127871419/226063488-b0532e48-409b-44d0-95e9-d0ec6fcce67e.png)

![image](https://user-images.githubusercontent.com/127871419/226063505-29b9924b-8ac3-4eae-a4ef-73253508f66d.png)




