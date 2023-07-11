import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from src.common import as_intrinsics_matrix, get_camera_from_tensor
from torch.utils.data import Dataset


def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if 'Y' not in header['channels'] else channelData['Y']

    return Y


def get_dataset(cfg, args, scale, device='cuda:0'):
    return dataset_dict[cfg['dataset']](cfg, args, scale, device=device)


class BaseDataset(Dataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(BaseDataset, self).__init__()
        self.name = cfg['dataset']
        self.device = device
        self.scale = scale
        self.png_depth_scale = cfg['cam']['png_depth_scale']

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']

        self.distortion = np.array(
            cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None
        self.crop_size = cfg['cam']['crop_size'] if 'crop_size' in cfg['cam'] else None

        if args.input_folder is None:
            self.input_folder = cfg['data']['input_folder']
        else:
            self.input_folder = args.input_folder

        self.crop_edge = cfg['cam']['crop_edge']
        # print("len base init:", len(self.color_paths))

    def __len__(self):
        return self.n_img

    def __getitem__(self, index):
        # print("index:", index)
        # print("len base get item:", len(self.color_paths))
        left_color_path = self.left_color_paths[index]
        right_color_path = self.right_color_paths[index]
        depth_path = self.depth_paths[index]
        left_color_data = cv2.imread(left_color_path)
        right_color_data = cv2.imread(right_color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            depth_data = readEXR_onlydepth(depth_path)
        
        # if self.distortion is not None:
        #     K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        #     # undistortion is only applied on color image, not depth!
        #     left_color_data = cv2.undistort(left_color_data, K, self.distortion)
        #     right_color_data = cv2.undistort(right_color_data, K, self.distortion)

        # Done in tracker and mapper
        # left_color_data = cv2.cvtColor(left_color_data, cv2.COLOR_BGR2RGB)
        # left_color_data = left_color_data / 255.
        # right_color_data = cv2.cvtColor(right_color_data, cv2.COLOR_BGR2RGB)
        # right_color_data = right_color_data / 255.
        # depth_data = depth_data.astype(np.float32) / self.png_depth_scale
        # H, W = depth_data.shape
        # left_color_data = cv2.resize(left_color_data, (W, H))
        # right_color_data = cv2.resize(right_color_data, (W, H))
        
        # Done in tracker and mapper
        left_color_data = torch.from_numpy(left_color_data)
        right_color_data = torch.from_numpy(right_color_data)

        depth_data = torch.from_numpy(depth_data.astype(float))
        # if self.crop_size is not None:
        #     # follow the pre-processing step in lietorch, actually is resize
        #     left_color_data = left_color_data.permute(2, 0, 1)
        #     left_color_data = F.interpolate(
        #         left_color_data[None], self.crop_size, mode='bilinear', align_corners=True)[0]
        #     left_color_data = left_color_data.permute(1, 2, 0).contiguous()
        #     right_color_data = right_color_data.permute(2, 0, 1)
        #     right_color_data = F.interpolate(
        #         right_color_data[None], self.crop_size, mode='bilinear', align_corners=True)[0]
        #     right_color_data = right_color_data.permute(1, 2, 0).contiguous()
        #     depth_data = F.interpolate(
        #         depth_data[None, None], self.crop_size, mode='nearest')[0, 0]


        # edge = self.crop_edge
        # if edge > 0:
        #     # crop image edge, there are invalid value on the edge of the color image
        #     right_color_data = right_color_data[edge:-edge, edge:-edge]
        #     left_color_data = left_color_data[edge:-edge, edge:-edge]
        #     depth_data = depth_data[edge:-edge, edge:-edge]
        pose = self.poses[index]
        pose[:3, 3] *= self.scale
        # print(type(color_data), type(depth_data), type(pose))
        return index, left_color_data.to(self.device), right_color_data.to(self.device), depth_data.to(self.device), pose.to(self.device)
        # return index, left_color_data, right_color_data, depth_data, pose.to(self.device)


class Replica(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(Replica, self).__init__(cfg, args, scale, device)
        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/results/frame*.jpg'))
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/results/depth*.png'))
        self.n_img = len(self.color_paths)
        self.load_poses(f'{self.input_folder}/traj.txt')

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class Azure(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(Azure, self).__init__(cfg, args, scale, device)
        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, 'color', '*.jpg')))
        self.depth_paths = sorted(
            glob.glob(os.path.join(self.input_folder, 'depth', '*.png')))
        self.n_img = len(self.color_paths)
        self.load_poses(os.path.join(
            self.input_folder, 'scene', 'trajectory.log'))

    def load_poses(self, path):
        self.poses = []
        if os.path.exists(path):
            with open(path) as f:
                content = f.readlines()

                # Load .log file.
                for i in range(0, len(content), 5):
                    # format %d (src) %d (tgt) %f (fitness)
                    data = list(map(float, content[i].strip().split(' ')))
                    ids = (int(data[0]), int(data[1]))
                    fitness = data[2]

                    # format %f x 16
                    c2w = np.array(
                        list(map(float, (''.join(
                            content[i + 1:i + 5])).strip().split()))).reshape((4, 4))

                    c2w[:3, 1] *= -1
                    c2w[:3, 2] *= -1
                    c2w = torch.from_numpy(c2w).float()
                    self.poses.append(c2w)
        else:
            for i in range(self.n_img):
                c2w = np.eye(4)
                c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w)


class ScanNet(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(ScanNet, self).__init__(cfg, args, scale, device)
        self.input_folder = os.path.join(self.input_folder, 'frames')
        # sorted() function is used to sort a sequence (e.g., a list, tuple, or string) in ascending order.
        # glob.glob() function searches for all files in a specified directory that match a specific pattern.
        self.color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        print("len scannet:", len(self.color_paths))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        # self.create_poses()
        self.load_poses(os.path.join(self.input_folder, 'pose'))
        self.n_img = len(self.color_paths)
    
    def create_poses(self):
        self.poses = []
        basic_matrix = np.array([[-0.930150, 0.243702, -0.274647, 2.598778],
                                [0.364940, 0.531076, -0.764707, 3.019463],
                                [-0.040502, -0.811522, -0.582916, 1.293332],
                                [0.000000, 0.000000, 0.000000, 1.000000]])
        basic_matrix = torch.from_numpy(basic_matrix).float()                         
        for _ in self.color_paths:
            # array_list = np.tile(basic_matrix, (self.n_img, 1, 1))
            # self.poses = array_list.tolist()
            self.poses.append(basic_matrix)

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class Highbay(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(Highbay, self).__init__(cfg, args, scale, device)
        self.input_folder = os.path.join(self.input_folder)
        # sorted() function is used to sort a sequence (e.g., a list, tuple, or string) in ascending order.
        # glob.glob() function searches for all files in a specified directory that match a specific pattern.
        self.left_color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'image_2', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.right_color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'image_3', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        # print((self.color_paths))
        # print("len high bay:", len(self.color_paths), len(self.depth_paths))
        if False: # assume we do not have camera pose
            self.create_poses()
        else:
            # if self.input_folder == 'Datasets/HighbayParkAll':
            self.load_poses_from_txt(os.path.join(self.input_folder, 'pose'))
            # else:
                # self.load_poses(os.path.join(self.input_folder, 'scene', 'trajectory.log'))
        self.n_img = len(self.left_color_paths)

    def create_poses(self):
        self.poses = []
        basic_matrix = np.array([[-0.930150, 0.243702, -0.274647, 2.598778],
                        [0.364940, 0.531076, -0.764707, 3.019463],
                        [-0.040502, -0.811522, -0.582916, 1.293332],
                        [0.000000, 0.000000, 0.000000, 1.000000]])
        # basic_matrix = np.array([[1,0,0,0],
        #                          [0,1,0,0],
        #                          [0,0,1,0],
        #                          [0,0,0,1]])
        basic_matrix = torch.from_numpy(basic_matrix).float()                         
        for _ in self.color_paths:
            # array_list = np.tile(basic_matrix, (self.n_img, 1, 1))
            # self.poses = array_list.tolist()
            self.poses.append(basic_matrix)

    def load_poses(self, path):
        self.poses = []
        if os.path.exists(path):
            with open(path) as f:
                content = f.readlines()

                # Load .log file.
                for i in range(0, len(content), 5):
                    # format %d (src) %d (tgt) %f (fitness)
                    data = list(map(float, content[i].strip().split(' ')))
                    ids = (int(data[0]), int(data[1]))
                    fitness = data[2]

                    # format %f x 16
                    c2w = np.array(
                        list(map(float, (''.join(
                            content[i + 1:i + 5])).strip().split()))).reshape((4, 4))

                    c2w[:3, 1] *= -1
                    c2w[:3, 2] *= -1
                    c2w = torch.from_numpy(c2w).float()
                    self.poses.append(c2w)
        else:
            for i in range(self.n_img):
                c2w = np.eye(4)
                c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w)
    
    #TODO:
    def load_poses_from_txt(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                file_lines = f.readlines()
            position_x, position_y, position_z = float(file_lines[10][8:]), float(file_lines[11][8:]), float(file_lines[12][8:])
            orientation_x, orientation_y, orientation_z, orientation_w = float(file_lines[14][8:]), float(file_lines[15][8:]), float(file_lines[16][8:]), float(file_lines[17][8:])
            # quad, T = inputs[:, :4], inputs[:, 4:]
            inputs = torch.tensor([orientation_x, orientation_y, orientation_z, orientation_w, position_x, position_y, position_z], device="cuda:0")
            c2w = get_camera_from_tensor(inputs).float().to('cpu')
            bottom = torch.tensor([0,0,0,1]).unsqueeze(0)
            c2w = torch.cat([c2w, bottom], dim=0)
            # c2w = np.array(ls).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            # c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)

    # def load_poses_ros_topic(self, path):
    #     # unfinished, just prepare to use directly from ros
    #     import rospy
    #     from nav_msgs.msg import Odometry

    #     # Read the ROS message from the .txt file
    #     with open('odom_data.txt', 'r') as f:
    #         message_str = f.read()

    #     odom_message = Odometry()
    #     odom_message.deserialize(message_str)

    #     # Extract the x, y, z, and w values from the orientation field in the pose
    #     x = odom_message.pose.pose.orientation.x
    #     y = odom_message.pose.pose.orientation.y
    #     z = odom_message.pose.pose.orientation.z
    #     w = odom_message.pose.pose.orientation.w


class Kitti(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(Kitti, self).__init__(cfg, args, scale, device)
        # self.input_folder = os.path.join(self.input_folder, 'frames')
        self.input_folder = self.input_folder
        # sorted() function is used to sort a sequence (e.g., a list, tuple, or string) in ascending order.
        # glob.glob() function searches for all files in a specified directory that match a specific pattern.
        self.color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'color', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        # print("self.color_paths len:", len(self.color_paths))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.n_img = len(self.color_paths)
        if True: # assume we do not have camera pose
            self.create_poses()
        else:
            self.load_poses(os.path.join(self.input_folder, 'pose'))

    def create_poses(self):
        self.poses = []
        basic_matrix = np.array([[1,0,0,0],
                                 [0,1,0,0],
                                 [0,0,1,0],
                                 [0,0,0,1]])
        basic_matrix = torch.from_numpy(basic_matrix).float()                         
        self.color_paths = sorted(glob.glob(os.path.join(self.input_folder, 'color', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        for _ in self.color_paths:
            # array_list = np.tile(basic_matrix, (self.n_img, 1, 1))
            # self.poses = array_list.tolist()
            self.poses.append(basic_matrix)

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class CoFusion(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(CoFusion, self).__init__(cfg, args, scale, device)
        self.input_folder = os.path.join(self.input_folder)
        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, 'colour', '*.png')))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth_noise', '*.exr')))
        self.n_img = len(self.color_paths)
        self.load_poses(os.path.join(self.input_folder, 'trajectories'))

    def load_poses(self, path):
        # We tried, but cannot align the coordinate frame of cofusion to ours.
        # So here we provide identity matrix as proxy.
        # But it will not affect the calculation of ATE since camera trajectories can be aligned.
        self.poses = []
        for i in range(self.n_img):
            c2w = np.eye(4)
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class TUM_RGBD(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(TUM_RGBD, self).__init__(cfg, args, scale, device)
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses += [c2w]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose


dataset_dict = {
    "replica": Replica,
    "scannet": ScanNet,
    "cofusion": CoFusion,
    "azure": Azure,
    "tumrgbd": TUM_RGBD,
    "kitti": Kitti,
    "highbay": Highbay
}
