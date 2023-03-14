from seg-xres-cam import vis_predict, seg_grad_cam, generate_masks, rise_segmentation, rise_aggregated

def visualize_algos(image, model, preprocess_transform = None, target = None, 
                    target_layer = None, box = None, DEVICE = 'cpu', 
                    method_indexes = [0], fig_base_name = None, fig_name = None, 
                    vis_base = True, vis = True, negative_gradient = False, 
                    pool_sizes = [], pool_modes = [], reshape_transformer = False,
                   n_masks = 0, input_size = None, p1 = 0.1, 
                    initial_mask_size = (7, 7), image_vis = None, vis_skip = 20, 
                    vis_rise = False, fig_size = None, grid = False):
    
    im, pred, rect = vis_predict(image, model, preprocess_transform = preprocess_transform,
                DEVICE = DEVICE, mask = None, box = box, vis = vis)
    
    results = [rect, pred]
    results_masks = []
    for i in range(len(method_indexes)):
        if len(pool_sizes) - 1 < i:
            pool_size = None
        else:
            pool_size = pool_sizes[i]
        if len(pool_modes) - 1 < i:
            pool_mode = np.max
        else:
            pool_mode = pool_modes[i]
        mask, overlay = seg_grad_cam(image, model, preprocess_transform = preprocess_transform,
                                     target = target, target_layer = target_layer,
                                     box = box, DEVICE = DEVICE, vis_base = vis_base,
                                     vis = vis, method_index = method_indexes[i], 
                                     reshape_transformer = reshape_transformer, 
                                    pool_size = pool_size, pool_mode = pool_mode)
        results.append(overlay)
        results_masks.append(mask)
    if n_masks != 0:
        masks = generate_masks(n_masks = n_masks, input_size = input_size, p1 = p1, initial_mask_size = initial_mask_size)
        coef = rise_segmentation(masks, image, model, preprocess_transform = preprocess_transform,
                                 target = target, box = box, DEVICE = DEVICE, vis = vis_rise, vis_skip = vis_skip)
        mask, overlay = rise_aggregated(image_vis, masks, coef, vis = vis_rise)
        results.append(overlay)
        results_masks.append(mask)
    if fig_size is not None:
        plt.figure(figsize = fig_size)
    else:
        plt.figure()
    for i in range(len(results)):
        plt.subplot(1, len(results), i + 1)
        plt.imshow(results[i])
        if grid == False:
            plt.axis('off')
    return results, results_masks
