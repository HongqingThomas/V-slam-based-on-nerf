# import numpy as np
# import torch
# import torch.nn.functional as F
# import cv2


# def as_intrinsics_matrix(intrinsics):
#     """
#     Get matrix representation of intrinsics.

#     """
#     K = np.eye(3)
#     K[0, 0] = intrinsics[0]
#     K[1, 1] = intrinsics[1]
#     K[0, 2] = intrinsics[2]
#     K[1, 2] = intrinsics[3]
#     return K


# def sample_pdf(bins, weights, N_samples, det=False, device='cuda:0'):
#     """
#     Hierarchical sampling in NeRF paper (section 5.2).

#     """
#     # Get pdf
#     weights = weights + 1e-5  # prevent nans
#     pdf = weights / torch.sum(weights, -1, keepdim=True)
#     cdf = torch.cumsum(pdf, -1)
#     # (batch, len(bins))
#     cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

#     # Take uniform samples
#     if det:
#         u = torch.linspace(0., 1., steps=N_samples)
#         u = u.expand(list(cdf.shape[:-1]) + [N_samples])
#     else:
#         u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

#     u = u.to(device)
#     # Invert CDF
#     u = u.contiguous()
#     try:
#         # this should work fine with the provided environment.yaml
#         inds = torch.searchsorted(cdf, u, right=True)
#     except:
#         # for lower version torch that does not have torch.searchsorted,
#         # you need to manually install from
#         # https://github.com/aliutkus/torchsearchsorted
#         from torchsearchsorted import searchsorted
#         inds = searchsorted(cdf, u, side='right')
#     below = torch.max(torch.zeros_like(inds-1), inds-1)
#     above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
#     inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

#     matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
#     cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
#     bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

#     denom = (cdf_g[..., 1]-cdf_g[..., 0])
#     denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
#     t = (u-cdf_g[..., 0])/denom
#     samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

#     return samples


# def random_select(l, k):
#     """
#     Random select k values from 0..l.

#     """
#     return list(np.random.permutation(np.array(range(l)))[:min(l, k)])


# def get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device):
#     """
#     Get corresponding rays from input uv.

#     """
#     if isinstance(c2w, np.ndarray):
#         c2w = torch.from_numpy(c2w).to(device)

#     dirs = torch.stack(
#         [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
#     dirs = dirs.reshape(-1, 1, 3)
#     # Rotate ray directions from camera frame to the world frame
#     # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     rays_d = torch.sum(dirs * c2w[:3, :3], -1)
#     rays_o = c2w[:3, -1].expand(rays_d.shape)
#     return rays_o, rays_d


# def select_uv(i, j, n, depth, color, device='cuda:0'):
#     """
#     Select n uv from dense uv.

#     """
#     i = i.reshape(-1)
#     j = j.reshape(-1)
#     indices = torch.randint(i.shape[0], (n,), device=device)
#     indices = indices.clamp(0, i.shape[0])
#     i = i[indices]  # (n)
#     j = j[indices]  # (n)
#     depth = depth.reshape(-1)
#     color = color.reshape(-1, 3)
#     depth = depth[indices]  # (n)
#     color = color[indices]  # (n,3)
#     return i, j, depth, color


# def get_sample_uv(H0, H1, W0, W1, n, depth, color, device='cuda:0'):
#     """
#     Sample n uv coordinates from an image region H0..H1, W0..W1

#     """
#     #TODO: Important
#     depth = depth[H0:H1, W0:W1]
#     color = color[H0:H1, W0:W1]
#     i, j = torch.meshgrid(torch.linspace(
#         W0, W1-1, W1-W0).to(device), torch.linspace(H0, H1-1, H1-H0).to(device))
    
#     i = i.t()  # transpose
#     j = j.t()
#     i, j, depth, color = select_uv(i, j, n, depth, color, device=device)
#     return i, j, depth, color


# def select_weighted_uv(i, j, n, depth, color, device='cuda:0'):
#     """
#     Select n uv from dense uv.

#     """
#     i = i.reshape(-1)
#     j = j.reshape(-1)
#     indices = torch.randint(i.shape[0], (n,), device=device)
#     indices = indices.clamp(0, i.shape[0])
#     i = i[indices]  # (n)
#     j = j[indices]  # (n)
#     depth = depth.reshape(-1)
#     color = color.reshape(-1, 3)
#     depth = depth[indices]  # (n)
#     color = color[indices]  # (n,3)
#     return i, j, depth, color


# def get_weighted_sample_uv(H0, H1, W0, W1, n, depth, color, device='cuda:0'):
#     """
#     Sample n uv coordinates from an image region H0..H1, W0..W1

#     """
#     #TODO: Important
#     depth = depth[H0:H1, W0:W1]
#     color = color[H0:H1, W0:W1]
#     weighted_map = gradient_filtered(color, depth, device)

#     i, j, depth, color = weighted_sample(H0, H1, W0, W1, n, depth, color, weighted_map, device=device)
#     return i, j, depth, color


# def gradient_filtered(color_img, depth_image, device):#test_realworld picture:150, 180 gazebo:230, 255 rosbag: 200,255
#     """
#     Apply sobel edge detection on input image in x, y direction
#     """
#     #1. Convert the image to gray scale
#     #2. Gaussian blur the image
#     #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
#     #4. Use cv2.addWeighted() to combine the results

#     ## TODO
#     color_img_cpu = color_img.detach().cpu().numpy()
#     color_img_cpu = np.float32(color_img_cpu)
#     gray_img = cv2.cvtColor(color_img_cpu, cv2.COLOR_RGB2GRAY)  #1
#     gray_blurred_img = cv2.GaussianBlur(gray_img,ksize=(5,5),sigmaX=0)  #2
#     grad_x = cv2.Sobel(gray_blurred_img, cv2.CV_64F,1,0,ksize=5)  #3
#     grad_y = cv2.Sobel(gray_blurred_img, cv2.CV_64F,0,1,ksize=5)
#     # abs_grad_x = cv2.convertScaleAbs(grad_x)
#     # abs_grad_y = cv2.convertScaleAbs(grad_y)
#     grad_color = cv2.addWeighted(grad_x, .5, grad_y, .5, 0)  #4
    
#     depth_img_cpu = depth_image.detach().cpu().numpy()
#     depth_img_cpu = np.float32(depth_img_cpu)
#     gray_blurred_img = cv2.GaussianBlur(depth_img_cpu,ksize=(5,5),sigmaX=0)  #2
#     grad_x = cv2.Sobel(gray_blurred_img, cv2.CV_64F,1,0,ksize=5)  #3
#     grad_y = cv2.Sobel(gray_blurred_img, cv2.CV_64F,0,1,ksize=5)
#     # abs_grad_x = cv2.convertScaleAbs(grad_x)
#     # abs_grad_y = cv2.convertScaleAbs(grad_y)
#     grad_depth = cv2.addWeighted(grad_x, .5, grad_y, .5, 0)  #4

#     grad = np.abs(np.multiply(grad_color, grad_depth))
#     grad = np.where(grad < 1, grad, 1 + np.log(grad))
#     grad_tensor = torch.tensor(grad).to(device)
#     return grad_tensor


# def weighted_sample(H0, H1, W0, W1, n, depth, color, weighted_map, device):
#         """
#         Description:
#             Perform resample to get a new list of particles 
#         params:
#             crop: H0, H1, W0, W1
#             sample: n
#             depth: tensor cropped depth
#             color: tensor cropped color
#             weighted_map: weighted map from copped depth & color (logged)
#         return:
#             i, j, depth, color: (n), (n), (n), (n, 3)
#         """
#         i, j = torch.meshgrid(torch.linspace(
#         W0, W1-1, W1-W0).to(device), torch.linspace(H0, H1-1, H1-H0).to(device))
#         i = i.t()  # transpose
#         j = j.t()
#         i = i.reshape(-1)
#         j = j.reshape(-1)

#         weighted_map = weighted_map.reshape(-1)
#         # print("weighted_map", weighted_map)
#         cum_weight = torch.cumsum(weighted_map, dim = 0)
#         total_weight = cum_weight[-1]
#         cum_weight = cum_weight/total_weight
#         random_index = torch.rand(n, device=device)
#         # indices = torch.zeros_like(random_index, dtype=int, device=device)

#         # for k in range (len(random_index)):
#         #     # print("cum_weight:", cum_weight)
#         #     # print("random_index:", random_index[i])
#         #     indices[k] = torch.where(cum_weight>random_index[k])[0][0]
#         _, indices = torch.max(cum_weight.unsqueeze(0) > random_index.unsqueeze(-1), dim=1)

#         # indices = torch.randint(i.shape[0], (n,), device=device)
#         # indices = indices.clamp(0, i.shape[0])
#         i = i[indices]  # (n)
#         j = j[indices]  # (n)
#         depth = depth.reshape(-1)
#         color = color.reshape(-1, 3)
#         depth = depth[indices]  # (n)
#         color = color[indices]  #

#         return i, j, depth, color


# def get_samples(H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, c2w, depth, color, device, flag):
#     """
#     Get n rays from the image region H0..H1, W0..W1.
#     c2w is its camera pose and depth/color is the corresponding image tensor.

#     """
#     if flag == 'mapper':
#         i, j, sample_depth, sample_color = get_sample_uv(
#             H0, H1, W0, W1, n, depth, color, device=device)
#     else:
#         i, j, sample_depth, sample_color = get_weighted_sample_uv(
#             H0, H1, W0, W1, n, depth, color, device=device)
#     rays_o, rays_d = get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device)
#     return rays_o, rays_d, sample_depth, sample_color


# def quad2rotation(quad):
#     """
#     Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

#     Args:
#         quad (tensor, batch_size*4): quaternion.

#     Returns:
#         rot_mat (tensor, batch_size*3*3): rotation.
#     """
#     bs = quad.shape[0]
#     qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
#     two_s = 2.0 / (quad * quad).sum(-1)
#     rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
#     rot_mat[:, 0, 0] = 1 - two_s * (qj ** 2 + qk ** 2)
#     rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
#     rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
#     rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
#     rot_mat[:, 1, 1] = 1 - two_s * (qi ** 2 + qk ** 2)
#     rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
#     rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
#     rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
#     rot_mat[:, 2, 2] = 1 - two_s * (qi ** 2 + qj ** 2)
#     return rot_mat


# def get_camera_from_tensor(inputs):
#     """
#     Convert quaternion and translation to transformation matrix.

#     """
#     N = len(inputs.shape)
#     if N == 1:
#         inputs = inputs.unsqueeze(0)
#     quad, T = inputs[:, :4], inputs[:, 4:]
#     R = quad2rotation(quad)
#     RT = torch.cat([R, T[:, :, None]], 2)
#     if N == 1:
#         RT = RT[0]
#     return RT


# def get_tensor_from_camera(RT, Tquad=False):
#     """
#     Convert transformation matrix to quaternion and translation.

#     """
#     gpu_id = -1
#     if type(RT) == torch.Tensor:
#         if RT.get_device() != -1:
#             RT = RT.detach().cpu()
#             gpu_id = RT.get_device()
#         RT = RT.numpy()
#     from mathutils import Matrix
#     R, T = RT[:3, :3], RT[:3, 3]
#     rot = Matrix(R)
#     quad = rot.to_quaternion()
#     if Tquad:
#         tensor = np.concatenate([T, quad], 0)
#     else:
#         tensor = np.concatenate([quad, T], 0)
#     tensor = torch.from_numpy(tensor).float()
#     if gpu_id != -1:
#         tensor = tensor.to(gpu_id)
#     return tensor


# def raw2outputs_nerf_color(raw, z_vals, rays_d, occupancy=False, device='cuda:0'):
#     """
#     Transforms model's predictions to semantically meaningful values.

#     Args:
#         raw (tensor, N_rays*N_samples*4): prediction from model.
#         z_vals (tensor, N_rays*N_samples): integration time.
#         rays_d (tensor, N_rays*3): direction of each ray.
#         occupancy (bool, optional): occupancy or volume density. Defaults to False.
#         device (str, optional): device. Defaults to 'cuda:0'.

#     Returns:
#         depth_map (tensor, N_rays): estimated distance to object.
#         depth_var (tensor, N_rays): depth variance/uncertainty.
#         rgb_map (tensor, N_rays*3): estimated RGB color of a ray.
#         weights (tensor, N_rays*N_samples): weights assigned to each sampled color.
#     """

#     def raw2alpha(raw, dists, act_fn=F.relu): return 1. - \
#         torch.exp(-act_fn(raw)*dists)
#     dists = z_vals[..., 1:] - z_vals[..., :-1]
#     dists = dists.float()
#     dists = torch.cat([dists, torch.Tensor([1e10]).float().to(
#         device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

#     # different ray angle corresponds to different unit length
#     dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
#     rgb = raw[..., :-1]
#     if occupancy:
#         raw[..., 3] = torch.sigmoid(10*raw[..., -1])
#         alpha = raw[..., -1]
#     else:
#         # original nerf, volume density
#         alpha = raw2alpha(raw[..., -1], dists)  # (N_rays, N_samples)

#     weights = alpha.float() * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(
#         device).float(), (1.-alpha + 1e-10).float()], -1).float(), -1)[:, :-1]
#     rgb_map = torch.sum(weights[..., None] * rgb, -2)  # (N_rays, 3)
#     depth_map = torch.sum(weights * z_vals, -1)  # (N_rays)
#     tmp = (z_vals-depth_map.unsqueeze(-1))  # (N_rays, N_samples)
#     depth_var = torch.sum(weights*tmp*tmp, dim=1)  # (N_rays)
#     return depth_map, depth_var, rgb_map, weights


# def get_rays(H, W, fx, fy, cx, cy, c2w, device):
#     """
#     Get rays for a whole image.

#     """
#     if isinstance(c2w, np.ndarray):
#         c2w = torch.from_numpy(c2w)
#     # pytorch's meshgrid has indexing='ij'
#     i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
#     i = i.t()  # transpose
#     j = j.t()
#     dirs = torch.stack(
#         [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
#     dirs = dirs.reshape(H, W, 1, 3)
#     # Rotate ray directions from camera frame to the world frame
#     # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     rays_d = torch.sum(dirs * c2w[:3, :3], -1)
#     rays_o = c2w[:3, -1].expand(rays_d.shape)
#     return rays_o, rays_d


# def normalize_3d_coordinate(p, bound):
#     """
#     Normalize coordinate to [-1, 1], corresponds to the bounding box given.

#     Args:
#         p (tensor, N*3): coordinate.
#         bound (tensor, 3*2): the scene bound.

#     Returns:
#         p (tensor, N*3): normalized coordinate.
#     """
#     p = p.reshape(-1, 3)
#     p[:, 0] = ((p[:, 0]-bound[0, 0])/(bound[0, 1]-bound[0, 0]))*2-1.0
#     p[:, 1] = ((p[:, 1]-bound[1, 0])/(bound[1, 1]-bound[1, 0]))*2-1.0
#     p[:, 2] = ((p[:, 2]-bound[2, 0])/(bound[2, 1]-bound[2, 0]))*2-1.0
#     return p

import numpy as np
import torch
import torch.nn.functional as F
import cv2


def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def sample_pdf(bins, weights, N_samples, det=False, device='cuda:0'):
    """
    Hierarchical sampling in NeRF paper (section 5.2).

    """
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    u = u.to(device)
    # Invert CDF
    u = u.contiguous()
    try:
        # this should work fine with the provided environment.yaml
        inds = torch.searchsorted(cdf, u, right=True)
    except:
        # for lower version torch that does not have torch.searchsorted,
        # you need to manually install from
        # https://github.com/aliutkus/torchsearchsorted
        from torchsearchsorted import searchsorted
        inds = searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


def random_select(l, k):
    """
    Random select k values from 0..l.

    """
    return list(np.random.permutation(np.array(range(l)))[:min(l, k)])


def get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device):
    """
    Get corresponding rays from input uv.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device)

    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(-1, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def select_uv(i, j, n, depth, color, device='cuda:0'):
    """
    Select n uv from dense uv.

    """
    i = i.reshape(-1)
    j = j.reshape(-1)
    indices = torch.randint(i.shape[0], (n,), device=device)
    indices = indices.clamp(0, i.shape[0])
    i = i[indices]  # (n)
    j = j[indices]  # (n)
    depth = depth.reshape(-1)
    color = color.reshape(-1, 3)
    depth = depth[indices]  # (n)
    color = color[indices]  # (n,3)
    return i, j, depth, color


def get_sample_uv(H0, H1, W0, W1, n, depth, color, device='cuda:0'):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1

    """
    #TODO: Important
    depth = depth[H0:H1, W0:W1]
    color = color[H0:H1, W0:W1]
    i, j = torch.meshgrid(torch.linspace(
        W0, W1-1, W1-W0).to(device), torch.linspace(H0, H1-1, H1-H0).to(device))
    
    i = i.t()  # transpose
    j = j.t()
    i, j, depth, color = select_uv(i, j, n, depth, color, device=device)
    return i, j, depth, color


def select_weighted_uv(i, j, n, depth, color, device='cuda:0'):
    """
    Select n uv from dense uv.

    """
    i = i.reshape(-1)
    j = j.reshape(-1)
    indices = torch.randint(i.shape[0], (n,), device=device)
    indices = indices.clamp(0, i.shape[0])
    i = i[indices]  # (n)
    j = j[indices]  # (n)
    depth = depth.reshape(-1)
    color = color.reshape(-1, 3)
    depth = depth[indices]  # (n)
    color = color[indices]  # (n,3)
    return i, j, depth, color


def get_weighted_sample_uv(random_index, H0, H1, W0, W1, n, depth, color, device='cuda:0'):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1

    """
    #TODO: Important
    depth = depth[H0:H1, W0:W1]
    color = color[H0:H1, W0:W1]
    weighted_map = gradient_filtered(color, depth, device)

    i, j, depth, color = weighted_sample(random_index, H0, H1, W0, W1, n, depth, color, weighted_map, device=device)
    return i, j, depth, color


def gradient_filtered(color_img, depth_image, device):#test_realworld picture:150, 180 gazebo:230, 255 rosbag: 200,255
    """
    Apply sobel edge detection on input image in x, y direction
    """
    #1. Convert the image to gray scale
    #2. Gaussian blur the image
    #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
    #4. Use cv2.addWeighted() to combine the results

    ## TODO
    color_img_cpu = color_img.detach().cpu().numpy()
    color_img_cpu = np.float32(color_img_cpu)
    gray_img = cv2.cvtColor(color_img_cpu, cv2.COLOR_RGB2GRAY)  #1
    gray_blurred_img = cv2.GaussianBlur(gray_img,ksize=(5,5),sigmaX=0)  #2
    grad_x = cv2.Sobel(gray_blurred_img, cv2.CV_64F,1,0,ksize=5)  #3
    grad_y = cv2.Sobel(gray_blurred_img, cv2.CV_64F,0,1,ksize=5)
    # abs_grad_x = cv2.convertScaleAbs(grad_x)
    # abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad_color = cv2.addWeighted(grad_x, .5, grad_y, .5, 0)  #4
    
    depth_img_cpu = depth_image.detach().cpu().numpy()
    depth_img_cpu = np.float32(depth_img_cpu)
    gray_blurred_img = cv2.GaussianBlur(depth_img_cpu,ksize=(5,5),sigmaX=0)  #2
    grad_x = cv2.Sobel(gray_blurred_img, cv2.CV_64F,1,0,ksize=5)  #3
    grad_y = cv2.Sobel(gray_blurred_img, cv2.CV_64F,0,1,ksize=5)
    # abs_grad_x = cv2.convertScaleAbs(grad_x)
    # abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad_depth = cv2.addWeighted(grad_x, .5, grad_y, .5, 0)  #4

    grad = np.abs(np.multiply(grad_color, grad_depth))
    grad = np.where(grad < 1, grad, 1 + np.log(grad))
    grad_tensor = torch.tensor(grad).to(device)
    return grad_tensor


def weighted_sample(random_index, H0, H1, W0, W1, n, depth, color, weighted_map, device):
        """
        Description:
            Perform resample to get a new list of particles 
        params:
            crop: H0, H1, W0, W1
            sample: n
            depth: tensor cropped depth
            color: tensor cropped color
            weighted_map: weighted map from copped depth & color (logged)
        return:
            i, j, depth, color: (n), (n), (n), (n, 3)
        """
        i, j = torch.meshgrid(torch.linspace(
        W0, W1-1, W1-W0).to(device), torch.linspace(H0, H1-1, H1-H0).to(device))
        i = i.t()  # transpose
        j = j.t()
        i = i.reshape(-1)
        j = j.reshape(-1)

        weighted_map = weighted_map.reshape(-1)
        # print("weighted_map", weighted_map)
        cum_weight = torch.cumsum(weighted_map, dim = 0)
        total_weight = cum_weight[-1]
        cum_weight = cum_weight/total_weight
        # random_index = torch.rand(n, device=device)
        # indices = torch.zeros_like(random_index, dtype=int, device=device)

        # for k in range (len(random_index)):
        #     # print("cum_weight:", cum_weight)
        #     # print("random_index:", random_index[i])
        #     indices[k] = torch.where(cum_weight>random_index[k])[0][0]
        _, indices = torch.max(cum_weight.unsqueeze(0) > random_index.unsqueeze(-1), dim=1)

        # indices = torch.randint(i.shape[0], (n,), device=device)
        # indices = indices.clamp(0, i.shape[0])
        i = i[indices]  # (n)
        j = j[indices]  # (n)
        depth = depth.reshape(-1)
        color = color.reshape(-1, 3)
        depth = depth[indices]  # (n)
        color = color[indices]  #

        return i, j, depth, color


def get_samples(random_index, H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, c2w, depth, color, device, flag):
    """
    Get n rays from the image region H0..H1, W0..W1.
    c2w is its camera pose and depth/color is the corresponding image tensor.

    """
    if flag == 'mapper':
        i, j, sample_depth, sample_color = get_sample_uv(
            H0, H1, W0, W1, n, depth, color, device=device)
    else:
        i, j, sample_depth, sample_color = get_weighted_sample_uv(random_index, 
            H0, H1, W0, W1, n, depth, color, device=device)
    rays_o, rays_d = get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device)
    return rays_o, rays_d, sample_depth, sample_color


def quad2rotation(quad):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    bs = quad.shape[0]
    qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    two_s = 2.0 / (quad * quad).sum(-1)
    rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
    rot_mat[:, 0, 0] = 1 - two_s * (qj ** 2 + qk ** 2)
    rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    rot_mat[:, 1, 1] = 1 - two_s * (qi ** 2 + qk ** 2)
    rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    rot_mat[:, 2, 2] = 1 - two_s * (qi ** 2 + qj ** 2)
    return rot_mat


def get_camera_from_tensor(inputs):
    """
    Convert quaternion and translation to transformation matrix.

    """
    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)
    quad, T = inputs[:, :4], inputs[:, 4:]
    R = quad2rotation(quad)
    RT = torch.cat([R, T[:, :, None]], 2)
    if N == 1:
        RT = RT[0]
    return RT


def get_tensor_from_camera(RT, Tquad=False):
    """
    Convert transformation matrix to quaternion and translation.

    """
    gpu_id = -1
    if type(RT) == torch.Tensor:
        if RT.get_device() != -1:
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    from mathutils import Matrix
    R, T = RT[:3, :3], RT[:3, 3]
    rot = Matrix(R)
    quad = rot.to_quaternion()
    if Tquad:
        tensor = np.concatenate([T, quad], 0)
    else:
        tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor


def raw2outputs_nerf_color(raw, z_vals, rays_d, device='cuda:0'):
    """
    Transforms model's predictions to semantically meaningful values.

    Args:
        raw (tensor, N_rays*N_samples*4): prediction from model.
        z_vals (tensor, N_rays*N_samples): integration time.
        rays_d (tensor, N_rays*3): direction of each ray.
        occupancy (bool, optional): occupancy or volume density. Defaults to False.
        device (str, optional): device. Defaults to 'cuda:0'.

    Returns:
        depth_map (tensor, N_rays): estimated distance to object.
        depth_var (tensor, N_rays): depth variance/uncertainty.
        rgb_map (tensor, N_rays*3): estimated RGB color of a ray.
        weights (tensor, N_rays*N_samples): weights assigned to each sampled color.
    """

    def raw2alpha(raw, dists, act_fn=F.relu): return 1. - \
        torch.exp(-act_fn(raw)*dists)
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = dists.float()
    dists = torch.cat([dists, torch.Tensor([1e10]).float().to(
        device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    # different ray angle corresponds to different unit length
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    color = raw[..., :-1]
    alpha = raw2alpha(raw[..., -1], dists)  # (N_rays, N_samples)

    weights = alpha.float() * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(
        device).float(), (1.-alpha + 1e-10).float()], -1).float(), -1)[:, :-1]
    color_map = torch.sum(weights[..., None] * color, -2)  # (N_rays, 3)
    depth_map = torch.sum(weights * z_vals, -1)  # (N_rays)
    tmp = (z_vals-depth_map.unsqueeze(-1))  # (N_rays, N_samples)
    depth_var = torch.sum(weights*(tmp)**2, dim=1)  # (N_rays)
    return depth_map, depth_var, color_map, weights


def get_rays(H, W, fx, fy, cx, cy, c2w, device):
    """
    Get rays for a whole image.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()  # transpose
    j = j.t()
    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(H, W, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def normalize_3d_coordinate(p, bound):
    """
    Normalize coordinate to [-1, 1], corresponds to the bounding box given.

    Args:
        p (tensor, N*3): coordinate.
        bound (tensor, 3*2): the scene bound.

    Returns:
        p (tensor, N*3): normalized coordinate.
    """
    p = p.reshape(-1, 3)
    p[:, 0] = ((p[:, 0]-bound[0, 0])/(bound[0, 1]-bound[0, 0]))*2-1.0
    p[:, 1] = ((p[:, 1]-bound[1, 0])/(bound[1, 1]-bound[1, 0]))*2-1.0
    p[:, 2] = ((p[:, 2]-bound[2, 0])/(bound[2, 1]-bound[2, 0]))*2-1.0
    return p



def depth_fusion(depth_data, predict_depth, depth_fusion_scale, device):
    '''
    description:
        params: depth_data(device tensor), predict_depth(device tensor)
        output: depth_data(device tensor)
    '''
    # print("depth_data :", depth_data.size(), torch.min(depth_data), torch.max(depth_data))
    # print("predict_depth :", predict_depth.size(), torch.min(predict_depth), torch.max(predict_depth))
    
    # depth_merge = torch.where(depth_data < predict_depth, depth_data, predict_depth)
    # depth_merge = depth_merge.to(device)
    # a = 1


    depth_data = depth_data.cpu().numpy()
    predict_depth = (predict_depth.cpu().numpy().astype(np.uint16))/1000.

    weight = 1 / (1 + np.exp(-1*depth_fusion_scale*(depth_data.astype(float)-20.)))
    merge_depth = (weight) * predict_depth + (1-weight) * depth_data

    # weight = 1 / (1 + torch.exp(-a*(depth_data.float()-20.))).to(device)
    # merge_depth = (weight) * predict_depth + (1-weight) * depth_data
    
    # print("depth_data: ",np.min(depth_data), np.max(depth_data))
    # print("predict_depth: ",np.min(predict_depth), np.max(predict_depth))
    # print("weight: ",np.min(weight), np.max(weight))
    # print("merge_depth: ",np.min(merge_depth), np.max(merge_depth))
    return torch.from_numpy(merge_depth).to(device)
    # return depth_data