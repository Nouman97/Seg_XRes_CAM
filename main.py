import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from seg-xres-cam import TorchSegmentationWrapper
from visualize import visualize_algos

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

n_masks, p1, window_size = 2000, 0.1, (7, 7)
input_size = image.shape
target_layer = model.model.backbone.layer4
method_indexes, pool_sizes, pool_modes, reshape_transformer = [0, 1, 1], [0, 1, 2], [None, np.mean, np.mean], False
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
