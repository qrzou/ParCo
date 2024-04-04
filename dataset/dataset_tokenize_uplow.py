import torch
from torch.utils import data

import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm

from dataset.dataset_VQ_uplow import whole2uplow, uplow2whole


class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias=5, unit_length=8, print_warning=False):
        # window_size should not be used in this dataset. It is only for training VQVAE.
        # self.window_size = window_size
        self.unit_length = unit_length
        self.feat_bias = feat_bias
        self.print_warning = print_warning

        self.dataset_name = dataset_name
        min_motion_len = 40 if dataset_name =='t2m' else 24
        
        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            #kinematic_chain = paramUtil.t2m_kinematic_chain
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            #kinematic_chain = paramUtil.kit_kinematic_chain
        
        joints_num = self.joints_num

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))
        
        split_file = pjoin(self.data_root, 'train.txt')
        
        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))

                # debug
                if np.isnan(motion).sum() > 0:
                    print('Detected NaN in Dataset, initialization stage!')
                    print('npy name:', pjoin(self.motion_dir, name + '.npy'))
                    raise Exception()

                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    if self.print_warning:
                        print('Skip the motion:', name, '. motion length is shorter than min_motion_len or greater than 200.')
                    continue

                data_dict[name] = {'motion': motion,
                                   'length': len(motion),
                                   'name': name}
                new_name_list.append(name)
                length_list.append(len(motion))

            except:
                if self.print_warning:
                    # Some motion may not exist in KIT dataset
                    print('Unable to load:', name)


        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list
        print(len(self.data_dict))

    def uplow2whole(self, uplow, mode='t2m', shared_joint_rec_mode='Avg'):
        rec_data = uplow2whole(uplow, mode, shared_joint_rec_mode)
        return rec_data

    def whole2uplow(self, motion, mode='t2m'):
        Upper_body, Lower_body = whole2uplow(motion, mode)
        return [Upper_body, Lower_body]

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']

        m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        uplow = self.whole2uplow(motion, mode=self.dataset_name)
        Upper_body, Lower_body  = uplow  # explicit written code for readability

        return Upper_body, Lower_body, name


def DATALoader(dataset_name,
               batch_size  =1,
               num_workers =8,
               unit_length =4) :
    
    train_loader = torch.utils.data.DataLoader(VQMotionDataset(dataset_name, unit_length=unit_length),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
