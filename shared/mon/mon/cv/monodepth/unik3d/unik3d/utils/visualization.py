import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image

from unik3d.utils.distributed import get_rank
from unik3d.utils.misc import ssi_helper


def colorize(
    value: np.ndarray, vmin: float = None, vmax: float = None, cmap: str = "magma_r"
):
    # if already RGB, do nothing
    if value.ndim > 2:
        if value.shape[-1] > 1:
            return value
        value = value[..., 0]
    invalid_mask = value < 0.0001
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # set color
    cmapper = plt.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 0
    img = value[..., :3]
    return img


def image_grid(imgs: list[np.ndarray], rows: int, cols: int) -> np.ndarray:
    if not len(imgs):
        return None
    assert len(imgs) == rows * cols
    h, w = imgs[0].shape[:2]
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(
            Image.fromarray(img.astype(np.uint8)).resize(
                (w, h), resample=Image.BILINEAR
            ),
            box=(i % cols * w, i // cols * h),
        )

    return np.array(grid)


def get_pointcloud_from_rgbd(
    image: np.array,
    depth: np.array,
    mask: np.ndarray,
    intrinsic_matrix: np.array,
    extrinsic_matrix: np.array = None,
):
    depth = np.array(depth).squeeze()
    mask = np.array(mask).squeeze()
    # Mask the depth array
    masked_depth = np.ma.masked_where(mask == False, depth)
    # masked_depth = np.ma.masked_greater(masked_depth, 8000)
    # Create idx array
    idxs = np.indices(masked_depth.shape)
    u_idxs = idxs[1]
    v_idxs = idxs[0]
    # Get only non-masked depth and idxs
    z = masked_depth[~masked_depth.mask]
    compressed_u_idxs = u_idxs[~masked_depth.mask]
    compressed_v_idxs = v_idxs[~masked_depth.mask]
    image = np.stack(
        [image[..., i][~masked_depth.mask] for i in range(image.shape[-1])], axis=-1
    )

    # Calculate local position of each point
    # Apply vectorized math to depth using compressed arrays
    cx = intrinsic_matrix[0, 2]
    fx = intrinsic_matrix[0, 0]
    x = (compressed_u_idxs - cx) * z / fx
    cy = intrinsic_matrix[1, 2]
    fy = intrinsic_matrix[1, 1]
    # Flip y as we want +y pointing up not down
    y = -((compressed_v_idxs - cy) * z / fy)

    # # Apply camera_matrix to pointcloud as to get the pointcloud in world coords
    # if extrinsic_matrix is not None:
    #     # Calculate camera pose from extrinsic matrix
    #     camera_matrix = np.linalg.inv(extrinsic_matrix)
    #     # Create homogenous array of vectors by adding 4th entry of 1
    #     # At the same time flip z as for eye space the camera is looking down the -z axis
    #     w = np.ones(z.shape)
    #     x_y_z_eye_hom = np.vstack((x, y, -z, w))
    #     # Transform the points from eye space to world space
    #     x_y_z_world = np.dot(camera_matrix, x_y_z_eye_hom)[:3]
    #     return x_y_z_world.T
    # else:
    x_y_z_local = np.stack((x, y, z), axis=-1)
    return np.concatenate([x_y_z_local, image], axis=-1)


def save_file_ply(xyz, rgb, pc_file):
    if rgb.max() < 1.001:
        rgb = rgb * 255.0
    rgb = rgb.astype(np.uint8)
    # print(rgb)
    with open(pc_file, "w") as f:
        # headers
        f.writelines(
            [
                "ply\n" "format ascii 1.0\n",
                "element vertex {}\n".format(xyz.shape[0]),
                "property float x\n",
                "property float y\n",
                "property float z\n",
                "property uchar red\n",
                "property uchar green\n",
                "property uchar blue\n",
                "end_header\n",
            ]
        )

        for i in range(xyz.shape[0]):
            str_v = "{:10.6f} {:10.6f} {:10.6f} {:d} {:d} {:d}\n".format(
                xyz[i, 0], xyz[i, 1], xyz[i, 2], rgb[i, 0], rgb[i, 1], rgb[i, 2]
            )
            f.write(str_v)


# really awful fct... FIXME


def train_artifacts(rgbs, gts, preds, infos={}):
    # interpolate to same shape, will be distorted! FIXME TODO
    shape = rgbs[0].shape[-2:]
    gts = F.interpolate(gts, shape, mode="nearest-exact")

    rgbs = [
        (127.5 * (rgb + 1))
        .clip(0, 255)
        .to(torch.uint8)
        .cpu()
        .detach()
        .permute(1, 2, 0)
        .numpy()
        for rgb in rgbs
    ]
    new_gts, new_preds = [], []
    num_additional, additionals = 0, []

    if len(gts) > 0:
        for i, gt in enumerate(gts):
            scale, shift = 1, 0
            up = torch.quantile(
                torch.log(1 + gts[i][gts[i] > 0]).float().cpu().detach(), 0.98
            ).item()
            down = torch.quantile(
                torch.log(1 + gts[i][gts[i] > 0]).float().cpu().detach(), 0.02
            ).item()
            gt = gts[i].cpu().detach().squeeze().numpy()
            pred = (preds[i].cpu().detach() * scale + shift).squeeze().numpy()
            new_gts.append(
                colorize(np.log(1.0 + gt), vmin=down, vmax=up)
            )  # , vmin=vmin, vmax=vmax))
            new_preds.append(
                colorize(np.log(1.0 + pred), vmin=down, vmax=up)
            )  # , vmin=vmin, vmax=vmax))

        gts, preds = new_gts, new_preds
    else:
        preds = [
            colorize(pred.cpu().detach().squeeze().numpy(), 0.0, 80.0)
            for i, pred in enumerate(preds)
        ]

    for name, info in infos.items():
        num_additional += 1
        if info.shape[1] == 3:
            additionals.extend(
                [
                    (127.5 * (x + 1))
                    .clip(0, 255)
                    .to(torch.uint8)
                    .cpu()
                    .detach()
                    .permute(1, 2, 0)
                    .numpy()
                    for x in info
                ]
            )
        else:  # must be depth!
            additionals.extend(
                [
                    colorize(x.cpu().detach().squeeze().numpy())
                    for i, x in enumerate(info)
                ]
            )

    num_rows = 2 + int(len(gts) > 0) + num_additional

    artifacts_grid = image_grid(
        [*rgbs, *gts, *preds, *additionals], num_rows, len(rgbs)
    )
    return artifacts_grid


def log_train_artifacts(rgbs, gts, preds, step, infos={}):
    artifacts_grid = train_artifacts(rgbs, gts, preds, infos)
    try:
        wandb.log({f"training": [wandb.Image(artifacts_grid)]}, step=step)
    except:
        Image.fromarray(artifacts_grid).save(
            os.path.join(
                os.environ.get("TMPDIR", "/tmp"),
                f"{get_rank()}_art_grid{step}.png",
            )
        )
        print("Logging training images failed")


def plot_quiver(flow, spacing, margin=0, **kwargs):
    """Plots less dense quiver field.

    Args:
        ax: Matplotlib axis
        flow: motion vectors
        spacing: space (px) between each arrow in grid
        margin: width (px) of enclosing region without arrows
        kwargs: quiver kwargs (default: angles="xy", scale_units="xy")
    """
    h, w, *_ = flow.shape

    nx = int((w - 2 * margin) / spacing)
    ny = int((h - 2 * margin) / spacing)

    x = np.linspace(margin, w - margin - 1, nx, dtype=np.int64)
    y = np.linspace(margin, h - margin - 1, ny, dtype=np.int64)

    flow = flow[np.ix_(y, x)]
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    kwargs = {**dict(angles="xy", scale_units="xy"), **kwargs}
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.quiver(x, y, u, v, **kwargs)

    return fig, ax
