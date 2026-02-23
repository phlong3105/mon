import numpy as np
import torch
from einops import rearrange

from unik3d.utils.camera import CameraSampler
from unik3d.utils.coordinate import coords_grid
from unik3d.utils.geometric import iou

try:
    from splatting import splatting_function
except Exception as e:
    splatting_function = None
    print(
        f"Splatting not available, please install it from github.com/hperrot/splatting"
    )


def fill(self, rgb, mask):
    def fill_noise(size, device):
        return torch.normal(0, 1.0, size=size, device=device)

    def fill_black(size, device):
        return -2 * torch.ones(size, device=device, dtype=torch.float32)

    def fill_white(size, device):
        return 2 * torch.ones(size, device=device, dtype=torch.float32)

    def fill_zero(size, device):
        return torch.zeros(size, device=device, dtype=torch.float32)

    B, C = rgb.shape[:2]
    validity_mask = mask.repeat(1, C, 1, 1).bool()
    for i in range(B):
        filler_fn = np.random.choice([fill_noise, fill_black, fill_white, fill_zero])
        rgb[i][~validity_mask[i]] = filler_fn(
            size=rgb[i][~validity_mask[i]].shape, device=rgb.device
        )
    return rgb


@torch.autocast(device_type="cuda", enabled=True, dtype=torch.float32)
def augment_camera(self, inputs, camera_sampler):
    rgb = inputs["image"]
    gt = inputs["depth"].clone()
    guidance = inputs[
        "depth_guidance"
    ]  # from GT if dense/synthetic or from a model's metric output
    validity_mask = inputs["validity_mask"].bool()
    dtype, device = gt.dtype, gt.device
    B, C, H, W = rgb.shape
    augmentable_indices = inputs["valid_camera"] & (
        inputs["depth_mask"].reshape(B, -1).float().mean(dim=1) > 0.0
    )

    augment_indices = torch.rand(B, 1, 1, device=device, dtype=dtype) > 0.9
    augment_indices[~augmentable_indices] = False
    id_coords = coords_grid(B, H, W, device=device)
    # get rescaled depth
    augment_indices = augment_indices.reshape(-1)
    for i, is_augment in enumerate(augment_indices):
        if not is_augment:
            continue

        pinhole_camera = inputs["camera"][i]
        fov = max(pinhole_camera.hfov[0], pinhole_camera.vfov[0]) * 180 / np.pi
        ratio = min(70.0 / fov, 1.0)  # decrease effect for larger fov
        if fov < 40.0:  # skips ~5%
            augment_indices[i] = False
            continue

        rgb_i = rgb[i : i + 1]
        id_coords_i = id_coords[i : i + 1]

        validity_mask_i = validity_mask[i : i + 1]
        depth = guidance[i : i + 1]

        if (depth < 0.0).any():
            augment_indices[i] = False
            continue

        depth = depth.sqrt()  # why sqrt??
        depth[~validity_mask_i] = depth.max() * 2.0

        fx, fy, cx, cy = pinhole_camera.params[:, :4].unbind(dim=-1)
        new_camera = camera_sampler(fx, fy, cx, cy, mult=1.0, ratio=ratio, H=H)
        unprojected = pinhole_camera.reconstruct(depth)
        projected = new_camera.project(unprojected)
        projection_mask = new_camera.projection_mask
        overlap_mask = (
            new_camera.overlap_mask
            if new_camera.overlap_mask is not None
            else torch.ones_like(projection_mask)
        )
        mask = validity_mask_i & overlap_mask

        # if it is actually going out, we need to remember the regions
        # remember when the tengetial distortion was keeping the validaty_mask border after re-warpingi
        # need a better way to define overlap class, in case of vortex style if will mask wrong parts...
        # also is_collapse does not take into consideration when we have vortex effect,
        # how can we avoid vortex in the first place????
        is_collapse = (projected[0, 1, 0, :] >= 0.0).all()
        if is_collapse:
            projected[~mask.repeat(1, 2, 1, 1)] = id_coords_i[~mask.repeat(1, 2, 1, 1)]
        flow = projected - id_coords_i
        depth[~mask] = depth.max() * 2.0

        if flow.norm(dim=1).median() / max(H, W) > 0.1:  # extreme cases
            augment_indices[i] = False
            continue

        # warp via soft splat
        depth_image = torch.cat([rgb_i, guidance[i : i + 1], mask], dim=1)
        depth_image = splatting_function(
            "softmax", depth_image, flow, -torch.log(1 + depth.clip(0.01))
        )
        rgb_warp = depth_image[:, :3]
        validity_mask_i = depth_image[:, -1:] > 0.0

        expanding = validity_mask_i.sum() > validity_mask[i : i + 1].sum()
        threshold = 0.7 if expanding else 0.25
        _iou = iou(validity_mask_i, validity_mask[i : i + 1])
        if _iou < threshold:  # too strong augmentation, lose most of the image
            augment_indices[i] = False
            continue

        # where it goes out
        mask_unwarpable = projection_mask & overlap_mask
        inputs["depth_mask"][i] = inputs["depth_mask"][i] & mask_unwarpable.squeeze(0)

        # compute new rays, and use the for supervision
        rays = new_camera.get_rays(shapes=(1, H, W))
        rays = rearrange(rays, "b c h w -> b (h w) c")
        inputs["rays"][i] = torch.where(
            rays.isnan().any(dim=-1, keepdim=True), 0.0, rays
        )[0]

        # update image, camera and validity_mask
        inputs["camera"][i] = new_camera
        inputs["image"][i] = self.fill(rgb_warp, validity_mask_i)[0]
        inputs["validity_mask"][i] = inputs["validity_mask"][i] & mask_unwarpable[0]

        # needed to reverse the augmentation for loss-computation (i.e. un-warp the prediction)
        inputs["grid_sample"][i] = projected[0]

    return inputs
