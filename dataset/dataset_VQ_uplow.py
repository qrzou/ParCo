import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm



class VQMotionDatasetUpLow(data.Dataset):
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

    def uplow2whole(self, uplow, mode='t2m', shared_joint_rec_mode='Avg'):
        rec_data = uplow2whole(uplow, mode, shared_joint_rec_mode)
        return rec_data

    def inv_transform(self, data):
        # de-normalization
        return data * self.std + self.mean
    
    def compute_sampling_prob(self):
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob

    def whole2uplow(self, motion, mode='t2m'):
        Upper_body, Lower_body = whole2uplow(motion, mode, window_size=self.window_size)
        return [Upper_body, Lower_body]

    def get_uplow_vel(self, uplow, mode='t2m'):
        uplow_vel_list = get_uplow_vel(uplow, mode=mode)
        return uplow_vel_list

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

        uplow = self.whole2uplow(motion, mode=self.dataset_name)

        Upper_body, Lower_body = uplow  # explicit written code for readability
        return [Upper_body, Lower_body]


def whole2uplow(motion, mode='t2m', window_size=None):
    # motion
    if mode == 't2m':

        # motion: raw_data (seg_len, joints_num, 3) ==> aug_data( seg_len-1, 263). seg_len-1: the last frame is dropped.
        aug_data = torch.from_numpy(motion)  # (nframes, 263)

        joints_num = 22
        s = 0  # start
        e = 4  # end
        root_data = aug_data[:, s:e]  # [seg_len-1, 4]
        s = e
        e = e + (joints_num - 1) * 3
        # ric_data = aug_data[:, 4 : (joints_num - 1) * 3 + 4]  # (joints_num - 1) means the 0th joint is dropped.
        ric_data = aug_data[:, s:e]  # [seq_len, (joints_num-1)*3]. (joints_num - 1) means the 0th joint is dropped.
        s = e
        e = e + (joints_num - 1) * 6
        # rot_data = aug_data[:, (joints_num - 1) * 3 + 4: (joint_num - 1) * 6 + (joints_num - 1) * 3 + 4]
        rot_data = aug_data[:, s:e]  # [seq_len, (joints_num-1) *6]
        s = e
        e = e + joints_num * 3
        local_vel = aug_data[:, s:e]  # [seq_len-1, joints_num*3]
        s = e
        e = e + 4
        feet = aug_data[:, s:e]  # [seg_len-1, 4]


        # remove the repeated 9-th joint in Upper Body and 0-th in Lower Body
        # remove the root (0-th joint) from the index
        # we process the root info individually
        Upper_idx = torch.Tensor([3, 6, 9, 12, 15,
                                  14, 17, 19, 21,
                                  13, 16, 18, 20]).to(torch.int64)  # Upper Body

        Lower_idx = torch.Tensor([3, 6, 9, 12, 15,
                                  2, 5, 8, 11,
                                  1, 4, 7, 10]).to(torch.int64)  # Lower Body

        root_idx = 0
        nframes = root_data.shape[0]
        if window_size is not None:
            assert nframes == window_size

        # The original shape of root_data and feet
        # root_data: (nframes, 4)
        # feet: (nframes, 4)
        ric_data = ric_data.reshape(nframes, -1, 3)    # (nframes, joints_num - 1, 3)
        rot_data = rot_data.reshape(nframes, -1, 6)    # (nframes, joints_num - 1, 6)
        local_vel = local_vel.reshape(nframes, -1, 3)  # (nframes, joints_num, 3)

        root_data = torch.cat([root_data, local_vel[:, root_idx, :]], dim=1)  # (nframes, 4+3=7)

        Upper = torch.cat([ric_data[:, Upper_idx - 1, :], rot_data[:, Upper_idx - 1, :], local_vel[:, Upper_idx, :]], dim=2)  # (nframes, 13, 3+6+3=12)
        Lower = torch.cat([ric_data[:, Lower_idx - 1, :], rot_data[:, Lower_idx - 1, :], local_vel[:, Lower_idx, :]], dim=2)  # (nframes, 13, 3+6+3=12)


        Root = root_data  # (nframes, 4+3=7)
        # feet: (nframes, 4)

        Upper_body = torch.cat([Upper.reshape(nframes, -1), Root], dim=1)        # (nframes, 13*12 + 7 = 163)
        Lower_body = torch.cat([Lower.reshape(nframes, -1), feet, Root], dim=1)  # (nframes, 13*12 + 4 + 7 = 167)

    elif mode == 'kit':
        raise NotImplementedError()

    else:
        raise Exception()

    return [Upper_body, Lower_body]


def uplow2whole(uplow, mode='t2m', shared_joint_rec_mode='Avg'):
    assert isinstance(uplow, list)

    if mode == 't2m':

        # UpLow to whole. (167, 163) ==> 263
        # we have:
        #   Upper_body (nframes, 163)
        #   Lower_body (nframes, 167)
        # we need to recover: root_data, ric_data, rot_data, local_vel, feet
        Upper_body, Lower_body = uplow


        if len(Upper_body.shape) == 3:  # (bs, nframes, upper_repre)
            bs = Upper_body.shape[0]
            nframes = Upper_body.shape[1]

        elif len(Upper_body.shape) == 2:
            bs = None
            nframes = Upper_body.shape[0]
        else:
            raise Exception()

        joints_num = 22
        device = Upper_body.device
        root_idx = 0

        Upper_Root = Upper_body[..., -7:]  # (B, nframes, 7) or (nframes, 7)
        Lower_Root = Lower_body[..., -7:]  # (B, nframes, 7) or (nframes, 7)

        if shared_joint_rec_mode == 'Avg':
            Root = (Upper_Root + Lower_Root) / 2
        elif shared_joint_rec_mode == 'Upper':
            Root = Upper_Root
        elif shared_joint_rec_mode == 'Lower':
            Root = Lower_Root
        else:
            raise Exception()

        rec_root_data = Root[..., :4]
        rec_feet = Lower_body[..., -11:-7]


        # remove the repeated 9-th joint in Upper Body and 0-th in Lower Body
        # remove the root (0-th joint) from the index
        # we process the root info individually
        Upper_idx = torch.Tensor([3, 6, 9, 12, 15,
                                  14, 17, 19, 21,
                                  13, 16, 18, 20]).to(torch.int64)  # Upper Body

        Lower_idx = torch.Tensor([3, 6, 9, 12, 15,
                                  2, 5, 8, 11,
                                  1, 4, 7, 10]).to(torch.int64)  # Lower Body


        if bs is None:
            Upper = Upper_body[..., :-7].reshape(nframes, 13, -1)   # (nframes, 13, 12)
            Lower = Lower_body[..., :-11].reshape(nframes, 13, -1)  # (nframes, 13, 12)

            # ric_data, rot_data, local_vel
            rec_ric_data = torch.zeros(nframes, joints_num-1, 3).to(device, dtype=rec_root_data.dtype)
            rec_rot_data = torch.zeros(nframes, joints_num-1, 6).to(device, dtype=rec_root_data.dtype)
            rec_local_vel = torch.zeros(nframes, joints_num, 3).to(device, dtype=rec_root_data.dtype)
            rec_local_vel[:, root_idx, :] = Root[:, 4:]

        else:
            Upper = Upper_body[..., :-7].reshape(bs, nframes, 13, -1)   # (B, nframes, 13, 12)
            Lower = Lower_body[..., :-11].reshape(bs, nframes, 13, -1)  # (B, nframes, 13, 12)

            # ric_data, rot_data, local_vel
            rec_ric_data = torch.zeros(bs, nframes, joints_num-1, 3).to(device, dtype=rec_root_data.dtype)
            rec_rot_data = torch.zeros(bs, nframes, joints_num-1, 6).to(device, dtype=rec_root_data.dtype)
            rec_local_vel = torch.zeros(bs, nframes, joints_num, 3).to(device, dtype=rec_root_data.dtype)
            rec_local_vel[..., root_idx, :] = Root[..., 4:]

        for part, idx in zip([Upper, Lower], [Upper_idx, Lower_idx]):
            rec_ric_data[..., idx - 1, :] = part[..., :, :3]
            rec_rot_data[..., idx - 1, :] = part[..., :, 3:9]
            rec_local_vel[..., idx, :] = part[..., :, 9:]

        # ########################
        # Choice of Backbone: Upper, Lower, or compute the mean
        # ########################
        Backbone_idx = torch.Tensor([3, 6, 9, 12, 15]).to(torch.int64)  # Backbone joints index

        # shared_joint_rec_mode is in ['Avg', 'Upper', 'Lower']
        if shared_joint_rec_mode == 'Upper':
            rec_ric_data[..., Backbone_idx - 1, :] = Upper[..., 0:5, :3]
            rec_rot_data[..., Backbone_idx - 1, :] = Upper[..., 0:5, 3:9]
            rec_local_vel[..., Backbone_idx, :] = Upper[..., 0:5, 9:]

        elif shared_joint_rec_mode == 'Lower':
            rec_ric_data[..., Backbone_idx - 1, :] = Lower[..., 0:5, :3]
            rec_rot_data[..., Backbone_idx - 1, :] = Lower[..., 0:5, 3:9]
            rec_local_vel[..., Backbone_idx, :] = Lower[..., 0:5, 9:]

        elif shared_joint_rec_mode == 'Avg':
            rec_ric_data[..., Backbone_idx - 1, :] = (Upper[..., 0:5, :3] + Lower[..., 0:5, :3]) / 2
            rec_rot_data[..., Backbone_idx - 1, :] = (Upper[..., 0:5, 3:9] + Lower[..., 0:5, 3:9]) / 2
            rec_local_vel[..., Backbone_idx, :] = (Upper[..., 0:5, 9:] + Lower[..., 0:5, 9:]) / 2

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
        raise NotImplementedError()

    else:
        raise Exception()

    return rec_data


def get_uplow_vel(uplow, mode='t2m'):
    assert isinstance(uplow, list)

    if mode == 't2m':

        # Extract each part's velocity from Upper and Lower body representation
        Upper_body, Lower_body = uplow

        if len(Upper_body.shape) == 3:  # (bs, nframes, part_repre)
            bs = Upper_body.shape[0]
            nframes = Upper_body.shape[1]

        elif len(Upper_body.shape) == 2:  # (nframes, part_repre)
            bs = None
            nframes = Upper_body.shape[0]

        else:
            raise Exception()

        if bs is None:
            Upper_Root = Upper_body[:, -7:]  # (nframes, 7)
            Lower_Root = Lower_body[:, -7:]  # (nframes, 7)

            upper_root_vel = Upper_Root[:, 4:].reshape(nframes, 1, -1)  # (nframes, 1, 3)
            lower_root_vel = Lower_Root[:, 4:].reshape(nframes, 1, -1)  # (nframes, 1, 3)

            Upper = Upper_body[:, :-7].reshape(nframes, 13, -1)   # (nframes, 13, 12)
            Lower = Lower_body[:, :-11].reshape(nframes, 13, -1)  # (nframes, 13, 12)

            Upper_vel = Upper[:, :, 9:]  # (nframes, 13, 3)
            Lower_vel = Lower[:, :, 9:]  # (nframes, 13, 3)

            Upper_vel = torch.cat([upper_root_vel, Upper_vel], dim=1)  # (nframes, 1+13=14, 3)
            Lower_vel = torch.cat([lower_root_vel, Lower_vel], dim=1)  # (nframes, 1+13=14, 3)

        else:
            Upper_Root = Upper_body[..., -7:]  # (B, nframes, 7)
            Lower_Root = Lower_body[..., -7:]  # (B, nframes, 7)

            upper_root_vel = Upper_Root[..., 4:].reshape(bs, nframes, 1, -1)  # (B, nframes, 1, 3)
            lower_root_vel = Lower_Root[..., 4:].reshape(bs, nframes, 1, -1)  # (B, nframes, 1, 3)

            Upper = Upper_body[..., :-7].reshape(bs, nframes, 13, -1)   # (B, nframes, 13, 12)
            Lower = Lower_body[..., :-11].reshape(bs, nframes, 13, -1)  # (B, nframes, 13, 12)

            Upper_vel = Upper[..., :, 9:]  # (B, nframes, 13, 3)
            Lower_vel = Lower[..., :, 9:]  # (B, nframes, 13, 3)

            Upper_vel = torch.cat([upper_root_vel, Upper_vel], dim=2)  # (B, nframes, 1+13=14, 3)
            Lower_vel = torch.cat([lower_root_vel, Lower_vel], dim=2)  # (B, nframes, 1+13=14, 3)

        uplow_vel_list = [Upper_vel, Lower_vel]


    elif mode == 'kit':
        raise NotImplementedError()


    else:
        raise Exception()

    return uplow_vel_list  # [Upper_vel, Lower_vel]




def DATALoader(dataset_name,
               batch_size,
               num_workers = 8,
               window_size = 64,
               unit_length = 4):
    
    trainSet = VQMotionDatasetUpLow(dataset_name, window_size=window_size, unit_length=unit_length)
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
