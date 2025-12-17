import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_geo_prior(img, geo_prior, save_path, batch_idx=0, point_coords=None, normalize=True, alpha=0.6):
    """
    Visualize geometric prior matrix and overlay the result on the original image
    Args:
        img: Original image tensor [B,C,H,W]
        geo_prior: Geometric prior tensor with shape [B,HW,HW]
        save_path: Save path
        batch_idx: Batch index to visualize
        point_coords: Reference point coordinates in format (h, w). If None, center point will be used
        normalize: Whether to normalize the display result
        alpha: Heatmap transparency, 0.0 means completely transparent, 1.0 means completely opaque
    """
    B, HW, _ = geo_prior.shape
    H = int(np.sqrt(HW))
    W = H  
    geo_prior_single = geo_prior[batch_idx]  # [HW,HW]
    
    if point_coords is None:
        center_h, center_w = H // 2, W // 2
        point_idx = center_h * W + center_w
    else:
        h, w = point_coords
        point_idx = h * W + w
    relation = geo_prior_single[point_idx]  # [HW]
    relation_map = relation.reshape(H, W)
    relation_np = relation_map.detach().cpu().numpy()
    
    if normalize:
        relation_np = (relation_np - relation_np.min()) / (relation_np.max() - relation_np.min() + 1e-6)
    
    orig_img = img[batch_idx].detach().cpu().numpy()
    orig_img = np.transpose(orig_img, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    orig_img = std * orig_img + mean
    orig_img = np.clip(orig_img * 255, 0, 255).astype(np.uint8)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    
    orig_h, orig_w = orig_img.shape[:2]
    
    colored_map = cv2.applyColorMap((relation_np * 255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
    colored_map = cv2.resize(colored_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    
    overlay = cv2.addWeighted(orig_img, 1-alpha, colored_map, alpha, 0)
    
    if point_coords is None:
        center_w_orig = int(center_w * orig_w / W)
        center_h_orig = int(center_h * orig_h / H)
        cv2.drawMarker(overlay, (center_w_orig, center_h_orig), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
    else:
        w_orig = int(w * orig_w / W)
        h_orig = int(h * orig_h / H)
        cv2.drawMarker(overlay, (w_orig, h_orig), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
    
    cv2.imwrite(save_path.replace('.png', '_overlay.png'), overlay)
    
    colored_map = cv2.applyColorMap((relation_np * 255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
    cv2.imwrite(save_path.replace('.png', '_heatmap.png'), colored_map)
    cv2.imwrite(save_path.replace('.png', '_original.png'), orig_img)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(relation_np, cmap='rainbow')
    plt.colorbar(label='Geometric Prior Strength')
    
    if point_coords is None:
        plt.plot(center_w, center_h, 'w*', markersize=10)
    else:
        plt.plot(w, h, 'w*', markersize=10)
    
    plt.title(f'Geometric Prior Visualization (Ref Point: {"center" if point_coords is None else f"({point_coords[0]}, {point_coords[1]})"})')
    plt.savefig(save_path)
    plt.close()
    
    return relation_map


def save_feature_visualization(feature_map, save_path):
    """
    Visualize feature map by averaging all feature maps into one image and resize to 518*518
    Args:
        feature_map: feature map tensor with shape [C,H,W]
        save_path: save path
    """
    
    if len(feature_map.shape) == 4:
        feature_map = feature_map.squeeze(0)
    mean_feature = torch.mean(feature_map, dim=0).detach().cpu().numpy()
    mean_feature = (mean_feature - mean_feature.min()) / (mean_feature.max() - mean_feature.min() + 1e-6)
    mean_feature = (mean_feature * 255).astype(np.uint8)
    mean_feature = cv2.resize(mean_feature, (518, 518), interpolation=cv2.INTER_LINEAR)
    
    colored_feature = cv2.applyColorMap(mean_feature, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(save_path, colored_feature)


def save_depth_visualization(depth_map, filename):
    """
    Save depth map visualization as a colored image.
    
    Args:
        depth_map (torch.Tensor): Depth map tensor with shape [H, W] or [B, H, W]
        filename (str): Output file path for the visualization
    """
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
    depth_norm = depth_norm.detach().cpu().numpy().astype(np.uint8)
    colored_depth = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    cv2.imwrite(filename, colored_depth)


def save_image(img_tensor, filename):
    """
    Save image tensor as a BGR image file.
    
    Args:
        img_tensor (torch.Tensor): Image tensor with shape [C, H, W] or [B, C, H, W]
        filename (str): Output file path for the image
    """
    img = img_tensor.detach().cpu().numpy()

    if img.shape[0] == 3:  
        img = np.transpose(img, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)
