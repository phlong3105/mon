#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Depth-Adapted Convolution layers.

References:
    - Paper: "Depth-Adapted CNN for RGB-D Cameras," ACCV 2021.
    - Code: https://github.com/Zongwei97/Depth-Adapted-CNN
"""

__all__ = [
    "compute_offset",
]

import torch


# ----- Utils -----
def grid(half_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the index for u and v direction for a grid.
    
    Args:
        half_size: The half size of the convolution filter.
    
    Returns:
        A tuple containing:
            - dir_u_index: The index for the u direction of the 2D filter.
            - dir_v_index: The index for the v direction of the 2D filter.
    """
    u_local = torch.arange(-half_size, half_size + 1)
    v_local = u_local
    patch_U, patch_V = torch.meshgrid(u_local, v_local, indexing='ij')
    dir_u_index      = patch_U.flatten()
    dir_v_index      = patch_V.flatten()
    return dir_u_index, dir_v_index


# ----- Step 1: Back-project -----
def neighbors2d(depth: torch.Tensor, conv_filter: int = 3, dilation: int = 1):
    """Compute the 2D neighbors for each pixel in the depth image.
    
    Args:
        depth: Depth image of shape ``(B, H, W)``.
        conv_filter: Size of the convolution filter ``(n x n)``. Default: ``3``.
        dilation: Dilation rate for the convolution filter. Default: ``1``.
    
    Returns:
        A tuple containing:
            - new: A tensor of shape ``(3, H', W', n*n, B)`` representing the 2D neighbors for each pixel.
            - dilated_posit: A tensor of shape ``(H', W', n*n, 2)`` representing the dilated positions of the neighbors.
    """
    device = depth.device
    h, w   = depth.shape[1], depth.shape[2]
    
    half_size     = int(conv_filter / 2)
    feature_map_h = h - conv_filter + 1
    feature_map_w = w - conv_filter + 1
    dir_u_index, dir_v_index = grid(half_size)
    dir_u_index   = dir_u_index.to(device)
    dir_v_index   = dir_v_index.to(device)
    
    new      = torch.zeros(depth.shape[0], conv_filter ** 2, feature_map_h, feature_map_w, 3).to(device)
    coord_V  = torch.arange(half_size + dir_v_index[0], half_size + feature_map_h + dir_v_index[0]).to(device)  # deplaced coord for the neighborhood
    coord_U  = torch.arange(half_size + dir_u_index[0], half_size + feature_map_w + dir_u_index[0]).to(device)
    test_UU, test_VV = torch.meshgrid(coord_U, coord_V)
    torch_UU = test_UU.T.to(device)
    torch_VV = test_VV.T.to(device)
    for i in range(conv_filter) :
        new[:, i::3,              :, :, 0] = torch_UU + i
        new[:, i * 3:(i + 1) * 3, :, :, 1] = torch_VV + i
    
    new[..., 2] = depth[:, new[..., 1].long(), new[..., 0].long()][:, 0, ...].to(device)
    new = new.permute(4, 2, 3, 1, 0).to(device)
    
    U_disp = torch.arange(half_size, half_size + feature_map_w).to(device)
    V_disp = torch.arange(half_size, half_size + feature_map_h).to(device)
    mat_coord_U_disp, mat_coord_V_disp = torch.meshgrid(U_disp, V_disp)
    mat_coord_U_disp    = mat_coord_U_disp.T.to(device)
    mat_coord_V_disp    = mat_coord_V_disp.T.to(device)
    center_filter_posit = torch.zeros(2,  feature_map_h, feature_map_w).to(device)
    center_filter_posit[0, :, :] = mat_coord_U_disp
    center_filter_posit[1, :, :] = mat_coord_V_disp
    dir_u_disp     = dilation * dir_u_index
    dir_v_disp     = dilation * dir_v_index
    grid_reg       = torch.zeros(len(dir_u_disp), 2).to(device)
    grid_reg[:, 0] = dir_u_disp
    grid_reg[:, 1] = dir_v_disp
    all_map        = torch.zeros(feature_map_h, feature_map_w, grid_reg.shape[0], grid_reg.shape[1]).to(device) + grid_reg
    dilated_posit  = center_filter_posit + all_map.permute(2, 3, 0, 1)
    return new, dilated_posit.permute(2, 3, 0, 1)


def camera_params():
    K  = torch.tensor(
        data=[[575.8157348632812, 0.0,               250],
              [0.0,               575.8157348632812, 250],
              [0.0,               0.0,               1.0]], dtype=torch.double)
    fw = K[0, 0]  # rapport f/ro_w
    fh = K[1, 1]
    u0 = K[0, 2]
    v0 = K[1, 2]
    # fw=320
    # fh=320
    # u0=320
    # v0=240
    mtx = torch.tensor([[fw, 0.0, u0], [0.0, fh, v0], [0.0, 0.0, 1.0]])
    return fw, fh, u0, v0 , mtx


def back_projection(neighbors_2d_posit: torch.Tensor) -> torch.Tensor:
    """Compute 3D point cloud of the neighbor for each pixel.
    
    Args:
        neighbors_2d_posit: A tensor of shape ``(3, H, W, n x n, B)`` representing the 2D neighbors for each pixel.
    
    Returns:
        A tensor of shape ``(3, H, W, n x n, B)`` representing the 3D positions of the neighbors.
    """
    device = neighbors_2d_posit.device
    fw, fh, u0, v0, _ = camera_params()
    fw = fw.to(device)
    fh = fh.to(device)
    u0 = u0.to(device)
    v0 = v0.to(device)
    u  = neighbors_2d_posit[0, :].to(device)
    v  = neighbors_2d_posit[1, :].to(device)
    z  = neighbors_2d_posit[2, :].to(device)
    neighbor_2d_posit_2_3d       = torch.zeros(neighbors_2d_posit.shape, dtype=torch.double).to(device)
    neighbor_2d_posit_2_3d[0, :] = (u - u0) / fw * z  # X
    neighbor_2d_posit_2_3d[1, :] = (v - v0) / fh * z  # Y
    neighbor_2d_posit_2_3d[2, :] = z                  # Z
    return neighbor_2d_posit_2_3d


# ----- Step 2: Plane fitting -----
def compute_plane(neighbors_2d_posit_2_3d: torch.Tensor) -> torch.Tensor:
    # Compute the normal (a, b, c, d) of the associated plane (least square) for each set of neighborhoods :  ax+by+cz+d = 0(supposed to be 1)
    # Input : 3D positions (3, h, w, n x n, batch)
    # Output : normal (batch, h, w, 4, 1)
    device  = neighbors_2d_posit_2_3d.device
    B       = neighbors_2d_posit_2_3d.shape[-1]
    H       = neighbors_2d_posit_2_3d.shape[1] # la hauteur de la matrice contenant les points_3D (matrice reduite)
    W       = neighbors_2d_posit_2_3d.shape[2]
    num_pts = neighbors_2d_posit_2_3d.shape[3]

    A = torch.zeros((B, H, W, num_pts, 3), dtype=torch.double).to(device)  # 2 colonnes
    A[:, :, :, :, 0] = neighbors_2d_posit_2_3d[0, :].permute(3, 0, 1, 2)
    A[:, :, :, :, 1] = neighbors_2d_posit_2_3d[1, :].permute(3, 0, 1, 2)
    A[:, :, :, :, 2] = 1
    b = torch.zeros((B, H, W, num_pts, 1), dtype=torch.double).to(device) # 1 colonne
    b[:, :, :, :, 0] = - neighbors_2d_posit_2_3d[2, :].permute(3, 0, 1, 2)

    A_transpose          = A.permute(0, 1, 2, 4, 3)  # A[:,:,:,i]  --- A_transpose[:,:,i,:]
    ata                  = torch.matmul(A_transpose, A).to(device)
    A_transpose_x_A_inv  = torch.inverse(ata).to(device)
    pseudo_inv_A         = torch.matmul(A_transpose_x_A_inv, A_transpose).to(device)
    abcd                 = torch.ones((B, pseudo_inv_A.shape[1], pseudo_inv_A.shape[2], pseudo_inv_A.shape[3] + 1, 1), dtype=torch.double).to(device)
    abcd[:, :, :, :2, :] = torch.matmul(pseudo_inv_A, b)[:, :, :, :2, :].to(device)
    abcd[:, :, :, -1, :] = torch.matmul(pseudo_inv_A, b)[:, :, :, -1, :].to(device)
    return abcd


# ----- Step 3: 3D planar grid -----
def orthogonal_projected_vectors(abcd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = abcd.device
    zu     = - abcd[..., 0, 0].to(device)
    u_3d   = torch.zeros((3, abcd.shape[1], abcd.shape[2], abcd.shape[0]), dtype=torch.float).to(device)
    v_3d   = torch.zeros((3, abcd.shape[1], abcd.shape[2], abcd.shape[0]), dtype=torch.float).to(device)
    z_3d   = torch.zeros((3, abcd.shape[1], abcd.shape[2], abcd.shape[0]), dtype=torch.float).to(device)
    z_3d[2, :] = 1
    
    u_3d[0, :] = 1
    u_3d[2, :] = zu.permute(1, 2, 0)
    u_3d       = u_3d / torch.norm(u_3d, dim=0)
    
    abc  = abcd[..., :3, 0].permute(3, 1, 2, 0).to(device)
    norm = (torch.norm(abc, dim=0) ** 2).to(device)
    u_3d = z_3d - abc / norm
    u_3d = u_3d / torch.norm(u_3d, dim=0)
    v_3d = torch.cross(abc, u_3d).to(device)
    v_3d = v_3d / torch.norm(v_3d, dim=0)
    
    return u_3d.unsqueeze(1), v_3d.unsqueeze(1)


def grid3d(
    neighbors_on_3d_plane: torch.Tensor,
    conv_filter          : int,
    u_proj               : torch.Tensor,
    v_proj               : torch.Tensor,
    depth_central_pixel  : torch.Tensor,
    bol                  : bool = False,
    dilation             : int  = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute the 3D grid for the convolution
    # Input : 3d point cloud with size (3, batch, h, w, n x n), convolution filter with size (n x n), u 3d vector with size (3, batch, h, w), v 3d vector with size (3, batch, h, w), associated depth of the central point on the image
    # Output : 3d regular grid on a local plane, size (n x n, 3, batch, h, w)
    device = neighbors_on_3d_plane.device
    u_proj = u_proj.to(device)
    v_proj = v_proj.to(device)
    fw, fh, _, _ , _= camera_params()
    
    point_3d_central = neighbors_on_3d_plane[:, :, :, :,int((conv_filter ** 2) / 2)].to(device)  # pt central est situe au milieu de la derniere dimension
    
    fv_indices, fu_indices = grid(int(conv_filter / 2))

    grid_reg       = torch.zeros((len(fv_indices), 2), dtype=torch.double).to(device)
    grid_reg[:, 0] = torch.tensor(fu_indices, dtype=torch.double)
    grid_reg[:, 1] = torch.tensor(fv_indices, dtype=torch.double)
    if bol:
        scale   = neighbors_on_3d_plane[2, :, :, :,int((conv_filter ** 2) / 2)].permute(1, 2, 0) / depth_central_pixel
        scale[scale < 1] = 1
        scale   = scale.to(device)
        u_scale = (dilation * depth_central_pixel * scale * u_proj).permute(1, 0, 2, 3, 4) / fw
        v_scale = (dilation * depth_central_pixel * scale * v_proj).permute(1, 0, 2, 3, 4) / fh
        # scale the norm, 40 is the nb of pixel when parallax
    else :
        u_scale = dilation * u_proj.permute(1, 0, 2, 3, 4) * depth_central_pixel / fw
        v_scale = dilation * v_proj.permute(1, 0, 2, 3, 4) * depth_central_pixel / fh
    # scale the norm, 40 is the nb of pixel when parallax
    
    # compute the norm for u3d and v3d. By convention, the depth of the central pixel is used.
    # We supposed there is virtual frontal-parallel plane which is used for the compute

    directions = torch.cat((u_scale, v_scale), dim=0).to(device)
    grid_3d    = point_3d_central + torch.einsum("ik,kj...->ij...", 1 * grid_reg, directions * 1).permute(0, 1, 4, 2, 3)
    
    return grid_3d, point_3d_central


def project_real_plane(
    neighbors_2d_posit_2_3d: torch.Tensor,
    abcd                   : torch.Tensor,
    conv_filter            : int,
    dilation               : int = 1
) -> torch.Tensor:
    neighbors_2d_posit_2_3d = neighbors_2d_posit_2_3d.permute(0, 4, 1, 2, 3)
    device  = neighbors_2d_posit_2_3d.device
    num_pts = neighbors_2d_posit_2_3d.shape[-1]
    center  = int(num_pts / 2)
    depth_central_pixel = neighbors_2d_posit_2_3d[2, :, :, :, center].mean(1).mean(1).to(device)
    u_3d, v_3d = orthogonal_projected_vectors(abcd)
    proj, _    = grid3d(
        neighbors_on_3d_plane = neighbors_2d_posit_2_3d,
        conv_filter           = conv_filter,
        u_proj                = u_3d,
        v_proj                = v_3d,
        depth_central_pixel   = depth_central_pixel,
        bol                   = True,
        dilation              = dilation,
    )
    return proj


# ----- Step 4: Project back to 2D -----
def grid_projection(grid_3d: torch.Tensor) -> torch.Tensor:
    # Compute the projection of 3d grid on the image
    # Input : 3d grid with size (n x n, 3, batch, h, w)
    # Output : 2d projection with size (n x n, 2, batch, h, w)
    device = grid_3d.device
    fw, fh, u0, v0, _ = camera_params()
    fw = fw.to(device)
    fh = fh.to(device)
    u0 = u0.to(device)
    v0 = v0.to(device)

    X  = grid_3d[:,0, :, :, :].to(device)
    Y  = grid_3d[:,1, :, :, :].to(device)
    Z  = grid_3d[:,2, :, :, :].to(device)
    
    grid_2d = torch.zeros((grid_3d.shape[0], 2, grid_3d.shape[2], grid_3d.shape[3], grid_3d.shape[4]), dtype=torch.double).to(device)
    grid_2d[:, 1, :] = X * fw / Z + u0  # u
    grid_2d[:, 0, :] = Y * fh / Z + v0  # v
    return grid_2d


# ------ Main Function -----
def compute_offset(depth: torch.Tensor, conv_filter: int = 3, dilation: int = 1) -> torch.Tensor:
    device = depth.device
    
    # Step 1: Back-project
    neighbors_2d_posit, dilated = neighbors2d(depth, conv_filter, dilation=dilation)
    neighbors_2d_posit_2_3d     = back_projection(neighbors_2d_posit).to(device)
    
    # Step 2: Plane fitting
    abcd = compute_plane(neighbors_2d_posit_2_3d).to(device)
    
    # Step 3: 3D planar grid
    grid_3d_reg = project_real_plane(neighbors_2d_posit_2_3d, abcd, conv_filter, dilation=dilation).to(device)
    
    # Step 4: Project back to 2D
    grid_2d = grid_projection(grid_3d_reg).to(device)
    ori     = dilated.to(device)
    inv_ori = torch.zeros(ori.shape).to(device)
    inv_ori[..., 0] = ori[..., 1]
    inv_ori[..., 1] = ori[..., 0]
    adapted     = grid_2d.permute(2, 3, 4, 0, 1).to(device)
    inv_adapted = torch.zeros(adapted.shape).to(device)
    for i in range(conv_filter):
        inv_adapted[:, :, :, i * conv_filter: (i + 1) * conv_filter] = adapted[:, :, :, i::conv_filter]
    
    # Step 5: Compute offset
    diff   = inv_adapted - inv_ori
    offset = diff.reshape(diff.shape[0], diff.shape[1], diff.shape[2], -1).to(device)
    offset = offset.permute(0, 3, 1, 2).to(device)
    return offset.float()


def compute_offset2(
    depth      : torch.Tensor,
    fx         : float,
    fy         : float,
    cx         : float,
    cy         : float,
    kernel_size: int = 3,
    dilation   : int = 1,
) -> torch.Tensor:
    # depth: (b, 1, h, w)
    # return: (b, 2*k*k, h, w)
    b, _, h, w = depth.shape
    device     = depth.device
    dtype      = depth.dtype

    k          = kernel_size
    half       = (k - 1) // 2
    a_range    = torch.arange(-half, half + 1, device=device, dtype=dtype) * dilation  # (k,)
    num_points = k * k

    # Pixel coordinates
    u, v = torch.meshgrid(torch.arange(w, device=device, dtype=dtype), torch.arange(h, device=device, dtype=dtype), indexing='xy')
    u = u.unsqueeze(0).expand(b, -1, -1)  # (b, h, w)
    v = v.unsqueeze(0).expand(b, -1, -1)

    # Center 3D
    Z0 = depth[:, 0]  # (b, h, w)
    X0 = (u - cx) * Z0 / fx
    Y0 = (v - cy) * Z0 / fy
    P0 = torch.stack([X0, Y0, Z0], dim=-1)  # (b, h, w, 3)

    # Neighbor offsets base
    du_base, dv_base = torch.meshgrid(a_range, a_range, indexing='ij')
    du_base = du_base.flatten()  # (k2,)
    dv_base = dv_base.flatten()

    # Neighbor pixels
    u_n = u.unsqueeze(-1) + du_base[None, None, None, :]  # (b, h, w, k2)
    v_n = v.unsqueeze(-1) + dv_base[None, None, None, :]

    # Clamp
    u_n = u_n.clamp(0, w - 1).long()
    v_n = v_n.clamp(0, h - 1).long()

    # Gather depths
    batch_idx = torch.arange(b, device=device).view(b, 1, 1, 1).expand(-1, h, w, num_points)
    Z_i  = depth[batch_idx, 0, v_n, u_n]  # (b, h, w, k2)

    # Neighbor 3D
    u_nf = u_n.float()
    v_nf = v_n.float()
    X_i  = (u_nf - cx) * Z_i / fx
    Y_i  = (v_nf - cy) * Z_i / fy
    P_i  = torch.stack([X_i, Y_i, Z_i], dim=-1)  # (b, h, w, k2, 3)

    # Centered
    Pc = P_i - P0.unsqueeze(-2)  # (b, h, w, k2, 3)

    # Flatten for SVD
    Pc_flat = Pc.reshape(b * h * w, num_points, 3)  # (bhw, k2, 3)

    # SVD
    _, _, Vh = torch.linalg.svd(Pc_flat, full_matrices=False)  # Vh: (bhw, 3, 3)

    # Normal n
    n = Vh[:, -1, :]  # (bhw, 3)
    n = n / (torch.norm(n, dim=-1, keepdim=True) + 1e-8)
    n1, n2, n3 = n.unbind(-1)

    # x_prime
    denom   = torch.sqrt(1 - n2 ** 2 + 1e-8)
    x1      = n3 / denom
    x2      = torch.zeros_like(n1)
    x3      = -n1 / denom
    x_prime = torch.stack([x1, x2, x3], dim=-1)  # (bhw, 3)

    # y_prime = n cross x_prime
    y1      = n2 * x3 - n3 * x2
    y2      = n3 * x1 - n1 * x3
    y3      = n1 * x2 - n2 * x1
    y_prime = torch.stack([y1, y2, y3], dim=-1)

    # Scale factors
    Zp = Z0.mean(dim=[1, 2])  # (b,)
    ku = dilation * Zp / fx  # (b,)
    kv = dilation * Zp / fy  # (b,)

    ku_flat = ku.view(b, 1, 1).expand(b, h, w).reshape(b * h * w)  # (bhw,)
    kv_flat = kv.view(b, 1, 1).expand(b, h, w).reshape(b * h * w)

    # 3D grid base
    aa_base, bb_base = torch.meshgrid(a_range, a_range, indexing='ij')  # (k, k)
    aa_base = aa_base.flatten()  # (k2,)
    bb_base = bb_base.flatten()

    # Scaled per point
    aa = aa_base[None, :] * ku_flat[:, None]  # (bhw, k2)
    bb = bb_base[None, :] * kv_flat[:, None]

    # R3D
    R3D = aa[..., None] * x_prime[:, None, :] + bb[..., None] * y_prime[:, None, :]  # (bhw, k2, 3)
    
    P0_flat = P0.reshape(b * h * w, 3)
    P_grid  = P0_flat[:, None, :] + R3D  # (bhw, k2, 3)
    
    Xg, Yg, Zg = P_grid.unbind(-1)  # (bhw, k2)

    ug = (Xg / (Zg + 1e-8)) * fx + cx
    vg = (Yg / (Zg + 1e-8)) * fy + cy

    u_flat = u.reshape(b * h * w)
    v_flat = v.reshape(b * h * w)

    du = ug - u_flat[:, None]
    dv = vg - v_flat[:, None]

    # Offset (bhw, k2, 2)
    offset_flat = torch.stack([du, dv], dim=-1)

    # Reshape to (b, 2*k2, h, w)
    offset = offset_flat.reshape(b, h, w, num_points, 2).permute(0, 3, 4, 1, 2).reshape(b, 2 * num_points, h, w)
    return offset


# ----- Debug -----
if __name__ == "__main__":
    depth  = torch.randn(1, 480, 640)
    # offset = compute_offset(depth, 3)
    # print(offset.shape)
    # print(offset)
    print("Start")
    offset2 = compute_offset2(depth.unsqueeze(1), 575.8157348632812, 575.8157348632812, 250, 250, 3)
    print(offset2.shape)
    print(offset2)
