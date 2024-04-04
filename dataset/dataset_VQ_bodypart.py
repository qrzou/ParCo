import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm



class VQMotionDatasetBodyPart(data.Dataset):
    def __init__(self, dataset_name, window_size=64, unit_length=4, print_warning=False):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name

        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'

        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21

            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        
        joints_num = self.joints_num

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))
        self.mean = mean
        self.std = std

        split_file = pjoin(self.data_root, 'train.txt')

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if motion.shape[0] < self.window_size:
                    if print_warning:
                        print('Skip the motion:', name, '. motion length shorter than window_size')
                    continue
                self.lengths.append(motion.shape[0] - self.window_size)
                self.data.append(motion)

            except:
                # Some motion may not exist in KIT dataset
                print('Unable to load:', name)

        print("Total number of motions {}".format(len(self.data)))

    def parts2whole(self, parts, mode='t2m', shared_joint_rec_mode='Avg'):
        rec_data = parts2whole(parts, mode, shared_joint_rec_mode)
        return rec_data

    def inv_transform(self, data):
        # de-normalization
        return data * self.std + self.mean
    
    def compute_sampling_prob(self):
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob

    def whole2parts(self, motion, mode='t2m'):
        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = whole2parts(motion, mode, window_size=self.window_size)
        return [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]

    def get_each_part_vel(self, parts, mode='t2m'):
        parts_vel_list = get_each_part_vel(parts, mode=mode)
        return parts_vel_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        motion = self.data[item]

        # Preprocess. We should set the slice of motion at getitem stage, not in the initialization.
        # If in the initialization, the augmentation of motion slice will be fixed, which will damage the diversity.
        idx = random.randint(0, len(motion) - self.window_size)
        motion = motion[idx:idx+self.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        parts = self.whole2parts(motion, mode=self.dataset_name)

        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts  # explicit written code for readability
        return [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]


def whole2parts(motion, mode='t2m', window_size=None):
    # motion
    if mode == 't2m':
        # 263-dims motion is actually an augmented motion representation
        # split the 263-dims data into the separated augmented data form:
        #    root_data, ric_data, rot_data, local_vel, feet
        aug_data = torch.from_numpy(motion)  # (nframes, 263)
        joints_num = 22
        s = 0  # start
        e = 4  # end
        root_data = aug_data[:, s:e]  # [seg_len-1, 4]
        s = e
        e = e + (joints_num - 1) * 3
        ric_data = aug_data[:, s:e]  # [seq_len, (joints_num-1)*3]. (joints_num - 1) means the 0th joint is dropped.
        s = e
        e = e + (joints_num - 1) * 6
        rot_data = aug_data[:, s:e]  # [seq_len, (joints_num-1) *6]
        s = e
        e = e + joints_num * 3
        local_vel = aug_data[:, s:e]  # [seq_len-1, joints_num*3]
        s = e
        e = e + 4
        feet = aug_data[:, s:e]  # [seg_len-1, 4]

        # move the root out of belowing parts
        R_L_idx = torch.Tensor([2, 5, 8, 11]).to(torch.int64)        # right leg
        L_L_idx = torch.Tensor([1, 4, 7, 10]).to(torch.int64)        # left leg
        B_idx = torch.Tensor([3, 6, 9, 12, 15]).to(torch.int64)      # backbone
        R_A_idx = torch.Tensor([9, 14, 17, 19, 21]).to(torch.int64)  # right arm
        L_A_idx = torch.Tensor([9, 13, 16, 18, 20]).to(torch.int64)  # left arm

        nframes = root_data.shape[0]
        if window_size is not None:
            assert nframes == window_size

        # The original shape of root_data and feet
        # root_data: (nframes, 4)
        # feet: (nframes, 4)
        ric_data = ric_data.reshape(nframes, -1, 3)    # (nframes, joints_num - 1, 3)
        rot_data = rot_data.reshape(nframes, -1, 6)    # (nframes, joints_num - 1, 6)
        local_vel = local_vel.reshape(nframes, -1, 3)  # (nframes, joints_num, 3)

        root_data = torch.cat([root_data, local_vel[:,0,:]], dim=1)  # (nframes, 4+3=7)
        R_L = torch.cat([ric_data[:, R_L_idx - 1, :], rot_data[:, R_L_idx - 1, :], local_vel[:, R_L_idx, :]], dim=2)  # (nframes, 4, 3+6+3=12)
        L_L = torch.cat([ric_data[:, L_L_idx - 1, :], rot_data[:, L_L_idx - 1, :], local_vel[:, L_L_idx, :]], dim=2)  # (nframes, 4, 3+6+3=12)
        B = torch.cat([ric_data[:, B_idx - 1, :], rot_data[:, B_idx - 1, :], local_vel[:, B_idx, :]], dim=2)  # (nframes, 5, 3+6+3=12)
        R_A = torch.cat([ric_data[:, R_A_idx - 1, :], rot_data[:, R_A_idx - 1, :], local_vel[:, R_A_idx, :]], dim=2)  # (nframes, 5, 3+6+3=12)
        L_A = torch.cat([ric_data[:, L_A_idx - 1, :], rot_data[:, L_A_idx - 1, :], local_vel[:, L_A_idx, :]], dim=2)  # (nframes, 5, 3+6+3=12)

        Root = root_data  # (nframes, 4+3=7)
        R_Leg = torch.cat([R_L.reshape(nframes, -1), feet[:, 2:]], dim=1)  # (nframes, 4*12+2=50)
        L_Leg = torch.cat([L_L.reshape(nframes, -1), feet[:, :2]], dim=1)  # (nframes, 4*12+2=50)
        Backbone = B.reshape(nframes, -1)  # (nframes, 5*12=60)
        R_Arm = R_A.reshape(nframes, -1)  # (nframes, 5*12=60)
        L_Arm = L_A.reshape(nframes, -1)  # (nframes, 5*12=60)

    elif mode == 'kit':
        # 251-dims motion is actually an augmented motion representation
        # split the 251-dims data into the separated augmented data form:
        #    root_data, ric_data, rot_data, local_vel, feet
        aug_data = torch.from_numpy(motion)  # (nframes, 251)
        joints_num = 21
        s = 0  # start
        e = 4  # end
        root_data = aug_data[:, s:e]  # [seg_len-1, 4]
        s = e
        e = e + (joints_num - 1) * 3
        ric_data = aug_data[:, s:e]  # [seq_len, (joints_num-1)*3]. (joints_num - 1) means the 0th joint is dropped.
        s = e
        e = e + (joints_num - 1) * 6
        rot_data = aug_data[:, s:e]  # [seq_len, (joints_num-1) *6]
        s = e
        e = e + joints_num * 3
        local_vel = aug_data[:, s:e]  # [seq_len-1, joints_num*3]
        s = e
        e = e + 4
        feet = aug_data[:, s:e]  # [seg_len-1, 4]

        # move the root joint 0-th out of belowing parts
        R_L_idx = torch.Tensor([11, 12, 13, 14, 15]).to(torch.int64)        # right leg
        L_L_idx = torch.Tensor([16, 17, 18, 19, 20]).to(torch.int64)        # left leg
        B_idx = torch.Tensor([1, 2, 3, 4]).to(torch.int64)      # backbone
        R_A_idx = torch.Tensor([3, 5, 6, 7]).to(torch.int64)  # right arm
        L_A_idx = torch.Tensor([3, 8, 9, 10]).to(torch.int64)  # left arm

        nframes = root_data.shape[0]
        if window_size is not None:
            assert nframes == window_size

        # The original shape of root_data and feet
        # root_data: (nframes, 4)
        # feet: (nframes, 4)
        ric_data = ric_data.reshape(nframes, -1, 3)    # (nframes, joints_num - 1, 3)
        rot_data = rot_data.reshape(nframes, -1, 6)    # (nframes, joints_num - 1, 6)
        local_vel = local_vel.reshape(nframes, -1, 3)  # (nframes, joints_num, 3)

        root_data = torch.cat([root_data, local_vel[:,0,:]], dim=1)  # (nframes, 4+3=7)
        R_L = torch.cat([ric_data[:, R_L_idx - 1, :], rot_data[:, R_L_idx - 1, :], local_vel[:, R_L_idx, :]], dim=2)  # (nframes, 4, 3+6+3=12)
        L_L = torch.cat([ric_data[:, L_L_idx - 1, :], rot_data[:, L_L_idx - 1, :], local_vel[:, L_L_idx, :]], dim=2)  # (nframes, 4, 3+6+3=12)
        B = torch.cat([ric_data[:, B_idx - 1, :], rot_data[:, B_idx - 1, :], local_vel[:, B_idx, :]], dim=2)  # (nframes, 5, 3+6+3=12)
        R_A = torch.cat([ric_data[:, R_A_idx - 1, :], rot_data[:, R_A_idx - 1, :], local_vel[:, R_A_idx, :]], dim=2)  # (nframes, 5, 3+6+3=12)
        L_A = torch.cat([ric_data[:, L_A_idx - 1, :], rot_data[:, L_A_idx - 1, :], local_vel[:, L_A_idx, :]], dim=2)  # (nframes, 5, 3+6+3=12)

        Root = root_data  # (nframes, 4+3=7)
        R_Leg = torch.cat([R_L.reshape(nframes, -1), feet[:, 2:]], dim=1)  # (nframes, 4*12+2=50)
        L_Leg = torch.cat([L_L.reshape(nframes, -1), feet[:, :2]], dim=1)  # (nframes, 4*12+2=50)
        Backbone = B.reshape(nframes, -1)  # (nframes, 5*12=60)
        R_Arm = R_A.reshape(nframes, -1)  # (nframes, 5*12=60)
        L_Arm = L_A.reshape(nframes, -1)  # (nframes, 5*12=60)

    else:
        raise Exception()

    return [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]


def parts2whole(parts, mode='t2m', shared_joint_rec_mode='Avg'):
    assert isinstance(parts, list)

    if mode == 't2m':
        # Parts to whole. (7, 50, 50, 60, 60, 60) ==> 263
        # we need to get root_data, ric_data, rot_data, local_vel, feet

        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts

        if len(Root.shape) == 3:  # (bs, nframes, part_repre)
            bs = Root.shape[0]
            nframes = Root.shape[1]

        elif len(Root.shape) == 2:
            bs = None
            nframes = Root.shape[0]
        else:
            raise Exception()

        joints_num = 22
        device = Root.device

        rec_root_data = Root[..., :4]
        rec_feet = torch.cat([L_Leg[..., -2:], R_Leg[..., -2:]], dim=-1)

        # move the root out of belowing parts
        R_L_idx = torch.Tensor([2, 5, 8, 11]).to(device, dtype=torch.int64)        # right leg
        L_L_idx = torch.Tensor([1, 4, 7, 10]).to(device, dtype=torch.int64)        # left leg
        B_idx = torch.Tensor([3, 6, 9, 12, 15]).to(device, dtype=torch.int64)      # backbone
        R_A_idx = torch.Tensor([9, 14, 17, 19, 21]).to(device, dtype=torch.int64)  # right arm
        L_A_idx = torch.Tensor([9, 13, 16, 18, 20]).to(device, dtype=torch.int64)  # left arm

        if bs is None:
            R_L = R_Leg[..., :-2].reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            L_L = L_Leg[..., :-2].reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            B = Backbone.reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            R_A = R_Arm.reshape(nframes, 5, -1)   # (nframes, 5, 3+6+3=12)
            L_A = L_Arm.reshape(nframes, 5, -1)   # (nframes, 5, 3+6+3=12)

            # ric_data, rot_data, local_vel
            rec_ric_data = torch.zeros(nframes, joints_num-1, 3).to(device, dtype=rec_root_data.dtype)
            rec_rot_data = torch.zeros(nframes, joints_num-1, 6).to(device, dtype=rec_root_data.dtype)
            rec_local_vel = torch.zeros(nframes, joints_num, 3).to(device, dtype=rec_root_data.dtype)
            rec_local_vel[:,0,:] = Root[:,4:]

        else:
            R_L = R_Leg[..., :-2].reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)
            L_L = L_Leg[..., :-2].reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)
            B = Backbone.reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)
            R_A = R_Arm.reshape(bs, nframes, 5, -1)   # (bs, nframes, 5, 3+6+3=12)
            L_A = L_Arm.reshape(bs, nframes, 5, -1)   # (bs, nframes, 5, 3+6+3=12)

            # ric_data, rot_data, local_vel
            rec_ric_data = torch.zeros(bs, nframes, joints_num-1, 3).to(device, dtype=rec_root_data.dtype)
            rec_rot_data = torch.zeros(bs, nframes, joints_num-1, 6).to(device, dtype=rec_root_data.dtype)
            rec_local_vel = torch.zeros(bs, nframes, joints_num, 3).to(device, dtype=rec_root_data.dtype)
            rec_local_vel[..., 0, :] = Root[..., 4:]

        for part, idx in zip([R_L, L_L, B, R_A, L_A], [R_L_idx, L_L_idx, B_idx, R_A_idx, L_A_idx]):
            # rec_ric_data[:, idx - 1, :] = part[:, :, :3]
            # rec_rot_data[:, idx - 1, :] = part[:, :, 3:9]
            # rec_local_vel[:, idx, :] = part[:, :, 9:]

            rec_ric_data[..., idx - 1, :] = part[..., :, :3]
            rec_rot_data[..., idx - 1, :] = part[..., :, 3:9]
            rec_local_vel[..., idx, :] = part[..., :, 9:]

        # ########################
        # Choose the origin of 9th joint, from B, R_A, L_A, or compute the mean
        # ########################
        idx = 9

        if shared_joint_rec_mode == 'L_Arm':
            rec_ric_data[..., idx - 1, :] = L_A[..., 0, :3]
            rec_rot_data[..., idx - 1, :] = L_A[..., 0, 3:9]
            rec_local_vel[..., idx, :] = L_A[..., 0, 9:]

        elif shared_joint_rec_mode == 'R_Arm':
            rec_ric_data[..., idx - 1, :] = R_A[..., 0, :3]
            rec_rot_data[..., idx - 1, :] = R_A[..., 0, 3:9]
            rec_local_vel[..., idx, :] = R_A[..., 0, 9:]

        elif shared_joint_rec_mode == 'Backbone':
            rec_ric_data[..., idx - 1, :] = B[..., 2, :3]
            rec_rot_data[..., idx - 1, :] = B[..., 2, 3:9]
            rec_local_vel[..., idx, :] = B[..., 2, 9:]

        elif shared_joint_rec_mode == 'Avg':
            rec_ric_data[..., idx - 1, :] = (L_A[..., 0, :3] + R_A[..., 0, :3] + B[..., 2, :3]) / 3
            rec_rot_data[..., idx - 1, :] = (L_A[..., 0, 3:9] + R_A[..., 0, 3:9] + B[..., 2, 3:9]) / 3
            rec_local_vel[..., idx, :] = (L_A[..., 0, 9:] + R_A[..., 0, 9:] + B[..., 2, 9:]) / 3

        else:
            raise Exception()

        # Concate them to 263-dims repre
        if bs is None:
            rec_ric_data = rec_ric_data.reshape(nframes, -1)
            rec_rot_data = rec_rot_data.reshape(nframes, -1)
            rec_local_vel = rec_local_vel.reshape(nframes, -1)

            rec_data = torch.cat([rec_root_data, rec_ric_data, rec_rot_data, rec_local_vel, rec_feet], dim=1)

        else:
            rec_ric_data = rec_ric_data.reshape(bs, nframes, -1)
            rec_rot_data = rec_rot_data.reshape(bs, nframes, -1)
            rec_local_vel = rec_local_vel.reshape(bs, nframes, -1)

            rec_data = torch.cat([rec_root_data, rec_ric_data, rec_rot_data, rec_local_vel, rec_feet], dim=2)

    elif mode == 'kit':

        # Parts to whole. (7, 62, 62, 48, 48, 48) ==> 251
        # we need to get root_data, ric_data, rot_data, local_vel, feet

        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts

        if len(Root.shape) == 3:  # (bs, nframes, part_repre)
            bs = Root.shape[0]
            nframes = Root.shape[1]

        elif len(Root.shape) == 2:
            bs = None
            nframes = Root.shape[0]
        else:
            raise Exception()

        joints_num = 21
        device = Root.device

        rec_root_data = Root[..., :4]
        rec_feet = torch.cat([L_Leg[..., -2:], R_Leg[..., -2:]], dim=-1)

        # move the root out of belowing parts
        R_L_idx = torch.Tensor([11, 12, 13, 14, 15]).to(device, dtype=torch.int64)  # right leg
        L_L_idx = torch.Tensor([16, 17, 18, 19, 20]).to(device, dtype=torch.int64)  # left leg
        B_idx = torch.Tensor([1, 2, 3, 4]).to(device, dtype=torch.int64)            # backbone
        R_A_idx = torch.Tensor([3, 5, 6, 7]).to(device, dtype=torch.int64)          # right arm
        L_A_idx = torch.Tensor([3, 8, 9, 10]).to(device, dtype=torch.int64)         # left arm

        if bs is None:
            R_L = R_Leg[..., :-2].reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            L_L = L_Leg[..., :-2].reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            B = Backbone.reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            R_A = R_Arm.reshape(nframes, 4, -1)   # (nframes, 4, 3+6+3=12)
            L_A = L_Arm.reshape(nframes, 4, -1)   # (nframes, 4, 3+6+3=12)

            # ric_data, rot_data, local_vel
            rec_ric_data = torch.zeros(nframes, joints_num-1, 3).to(device, dtype=rec_root_data.dtype)
            rec_rot_data = torch.zeros(nframes, joints_num-1, 6).to(device, dtype=rec_root_data.dtype)
            rec_local_vel = torch.zeros(nframes, joints_num, 3).to(device, dtype=rec_root_data.dtype)
            rec_local_vel[:,0,:] = Root[:,4:]

        else:
            R_L = R_Leg[..., :-2].reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)
            L_L = L_Leg[..., :-2].reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)
            B = Backbone.reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)
            R_A = R_Arm.reshape(bs, nframes, 4, -1)   # (bs, nframes, 4, 3+6+3=12)
            L_A = L_Arm.reshape(bs, nframes, 4, -1)   # (bs, nframes, 4, 3+6+3=12)

            # ric_data, rot_data, local_vel
            rec_ric_data = torch.zeros(bs, nframes, joints_num-1, 3).to(device, dtype=rec_root_data.dtype)
            rec_rot_data = torch.zeros(bs, nframes, joints_num-1, 6).to(device, dtype=rec_root_data.dtype)
            rec_local_vel = torch.zeros(bs, nframes, joints_num, 3).to(device, dtype=rec_root_data.dtype)
            rec_local_vel[..., 0, :] = Root[..., 4:]

        for part, idx in zip([R_L, L_L, B, R_A, L_A], [R_L_idx, L_L_idx, B_idx, R_A_idx, L_A_idx]):

            rec_ric_data[..., idx - 1, :] = part[..., :, :3]
            rec_rot_data[..., idx - 1, :] = part[..., :, 3:9]
            rec_local_vel[..., idx, :] = part[..., :, 9:]

        # ########################
        # Choose the origin of 3-th joint, from B, R_A, L_A, or compute the mean
        # ########################
        idx = 3

        if shared_joint_rec_mode == 'L_Arm':
            rec_ric_data[..., idx - 1, :] = L_A[..., 0, :3]
            rec_rot_data[..., idx - 1, :] = L_A[..., 0, 3:9]
            rec_local_vel[..., idx, :] = L_A[..., 0, 9:]

        elif shared_joint_rec_mode == 'R_Arm':
            rec_ric_data[..., idx - 1, :] = R_A[..., 0, :3]
            rec_rot_data[..., idx - 1, :] = R_A[..., 0, 3:9]
            rec_local_vel[..., idx, :] = R_A[..., 0, 9:]

        elif shared_joint_rec_mode == 'Backbone':
            rec_ric_data[..., idx - 1, :] = B[..., 2, :3]
            rec_rot_data[..., idx - 1, :] = B[..., 2, 3:9]
            rec_local_vel[..., idx, :] = B[..., 2, 9:]

        elif shared_joint_rec_mode == 'Avg':
            rec_ric_data[..., idx - 1, :] = (L_A[..., 0, :3] + R_A[..., 0, :3] + B[..., 2, :3]) / 3
            rec_rot_data[..., idx - 1, :] = (L_A[..., 0, 3:9] + R_A[..., 0, 3:9] + B[..., 2, 3:9]) / 3
            rec_local_vel[..., idx, :] = (L_A[..., 0, 9:] + R_A[..., 0, 9:] + B[..., 2, 9:]) / 3

        else:
            raise Exception()

        # Concate them to 251-dims repre
        if bs is None:
            rec_ric_data = rec_ric_data.reshape(nframes, -1)
            rec_rot_data = rec_rot_data.reshape(nframes, -1)
            rec_local_vel = rec_local_vel.reshape(nframes, -1)

            rec_data = torch.cat([rec_root_data, rec_ric_data, rec_rot_data, rec_local_vel, rec_feet], dim=1)

        else:
            rec_ric_data = rec_ric_data.reshape(bs, nframes, -1)
            rec_rot_data = rec_rot_data.reshape(bs, nframes, -1)
            rec_local_vel = rec_local_vel.reshape(bs, nframes, -1)

            rec_data = torch.cat([rec_root_data, rec_ric_data, rec_rot_data, rec_local_vel, rec_feet], dim=2)

    else:
        raise Exception()

    return rec_data


def get_each_part_vel(parts, mode='t2m'):
    assert isinstance(parts, list)

    if mode == 't2m':
        # Extract each part's velocity from parts representation
        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts

        if len(Root.shape) == 3:  # (bs, nframes, part_repre)
            bs = Root.shape[0]
            nframes = Root.shape[1]

        elif len(Root.shape) == 2:  # (nframes, part_repre)
            bs = None
            nframes = Root.shape[0]

        else:
            raise Exception()

        Root_vel = Root[..., 4:]
        if bs is None:
            R_L = R_Leg[:, :-2].reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            L_L = L_Leg[:, :-2].reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            B = Backbone.reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            R_A = R_Arm.reshape(nframes, 5, -1)   # (nframes, 5, 3+6+3=12)
            L_A = L_Arm.reshape(nframes, 5, -1)   # (nframes, 5, 3+6+3=12)

            R_Leg_vel = R_L[:, :, 9:].reshape(nframes, -1)
            L_Leg_vel = L_L[:, :, 9:].reshape(nframes, -1)
            Backbone_vel = B[:, :, 9:].reshape(nframes, -1)
            R_Arm_vel = R_A[:, :, 9:].reshape(nframes, -1)
            L_Arm_vel = L_A[:, :, 9:].reshape(nframes, -1)

        else:
            R_L = R_Leg[:, :, :-2].reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)
            L_L = L_Leg[:, :, :-2].reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)
            B = Backbone.reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)
            R_A = R_Arm.reshape(bs, nframes, 5, -1)   # (bs, nframes, 5, 3+6+3=12)
            L_A = L_Arm.reshape(bs, nframes, 5, -1)   # (bs, nframes, 5, 3+6+3=12)

            R_Leg_vel = R_L[:, :, :, 9:].reshape(bs, nframes, -1)  # (bs, nframes, nb_joints, 3) ==> (bs, nframes, vel_dim)
            L_Leg_vel = L_L[:, :, :, 9:].reshape(bs, nframes, -1)
            Backbone_vel = B[:, :, :, 9:].reshape(bs, nframes, -1)
            R_Arm_vel = R_A[:, :, :, 9:].reshape(bs, nframes, -1)
            L_Arm_vel = L_A[:, :, :, 9:].reshape(bs, nframes, -1)

        parts_vel_list = [Root_vel, R_Leg_vel, L_Leg_vel, Backbone_vel, R_Arm_vel, L_Arm_vel]

    elif mode == 'kit':
        # Extract each part's velocity from parts representation
        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts

        if len(Root.shape) == 3:  # (bs, nframes, part_repre)
            bs = Root.shape[0]
            nframes = Root.shape[1]

        elif len(Root.shape) == 2:  # (nframes, part_repre)
            bs = None
            nframes = Root.shape[0]

        else:
            raise Exception()

        Root_vel = Root[..., 4:]
        if bs is None:
            R_L = R_Leg[:, :-2].reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            L_L = L_Leg[:, :-2].reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            B = Backbone.reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            R_A = R_Arm.reshape(nframes, 4, -1)   # (nframes, 4, 3+6+3=12)
            L_A = L_Arm.reshape(nframes, 4, -1)   # (nframes, 4, 3+6+3=12)

            R_Leg_vel = R_L[:, :, 9:].reshape(nframes, -1)
            L_Leg_vel = L_L[:, :, 9:].reshape(nframes, -1)
            Backbone_vel = B[:, :, 9:].reshape(nframes, -1)
            R_Arm_vel = R_A[:, :, 9:].reshape(nframes, -1)
            L_Arm_vel = L_A[:, :, 9:].reshape(nframes, -1)

        else:
            R_L = R_Leg[:, :, :-2].reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)
            L_L = L_Leg[:, :, :-2].reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)
            B = Backbone.reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)
            R_A = R_Arm.reshape(bs, nframes, 4, -1)   # (bs, nframes, 4, 3+6+3=12)
            L_A = L_Arm.reshape(bs, nframes, 4, -1)   # (bs, nframes, 4, 3+6+3=12)

            R_Leg_vel = R_L[:, :, :, 9:].reshape(bs, nframes, -1)  # (bs, nframes, nb_joints, 3) ==> (bs, nframes, vel_dim)
            L_Leg_vel = L_L[:, :, :, 9:].reshape(bs, nframes, -1)
            Backbone_vel = B[:, :, :, 9:].reshape(bs, nframes, -1)
            R_Arm_vel = R_A[:, :, :, 9:].reshape(bs, nframes, -1)
            L_Arm_vel = L_A[:, :, :, 9:].reshape(bs, nframes, -1)

        parts_vel_list = [Root_vel, R_Leg_vel, L_Leg_vel, Backbone_vel, R_Arm_vel, L_Arm_vel]

    else:
        raise Exception()

    return parts_vel_list  # [Root_vel, R_Leg_vel, L_Leg_vel, Backbone_vel, R_Arm_vel, L_Arm_vel]




def DATALoader(dataset_name,
               batch_size,
               num_workers = 8,
               window_size = 64,
               unit_length = 4):
    
    trainSet = VQMotionDatasetBodyPart(dataset_name, window_size=window_size, unit_length=unit_length)
    prob = trainSet.compute_sampling_prob()
    sampler = torch.utils.data.WeightedRandomSampler(prob, num_samples = len(trainSet) * 1000, replacement=True)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size,
                                              shuffle=True,
                                              #sampler=sampler,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
