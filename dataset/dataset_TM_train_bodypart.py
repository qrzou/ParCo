from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm

import torch
# from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
import numpy as np

import utils.paramUtil as paramUtil


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(Dataset):
    def __init__(self, dataset_name, vq_dir, unit_length, codebook_size, print_warning=False):
        
        self.max_length = 64
        self.pointer = 0
        self.dataset_name = dataset_name
        self.print_warning = print_warning

        self.unit_length = unit_length
        # self.mot_start_idx = codebook_size
        self.mot_end_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1
        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 26 if unit_length == 8 else 51
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain

        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 26 if unit_length == 8 else 51
            kinematic_chain = paramUtil.kit_kinematic_chain

        self.vq_dir = vq_dir
        self.parts_name = ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']
        split_file = pjoin(self.data_root, 'train.txt')


        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list):
            try:
                # Load the tokenized motion.
                # Training GPT only need to concern the tokens of motion.
                m_token_dict = {}
                for p_n in self.parts_name:
                    # m_token_list = np.load(pjoin(vq_dir, name + '_' + p_n + '.npy'))
                    m_token = np.load(pjoin(vq_dir, name + '_' + p_n + '.npy'))
                    m_token_dict[p_n] = m_token

                # m_token_list = np.load(pjoin(vq_dir, '%s.npy'%name))

                # Read text
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()

                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                # Create a new motion token dict for the motion sliced from the original motion.
                                m_token_dict_new = {}
                                m_token_len = []
                                for p_n in self.parts_name:
                                    # m_token: numpy.array, (1, token_len).
                                    #  Thus the following code can filter the tokens with wrong [f_tag, to_tag].
                                    m_token = m_token_dict[p_n]
                                    m_token_new = [tokens[int(f_tag*fps/unit_length) : int(to_tag*fps/unit_length)] for tokens in m_token if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length)]
                                    # Debug: the original code is from T2M-GPT,
                                    #   but there seems no need for putting just one m_token in a list.
                                    if len(m_token_new) > 1:
                                        print('Detect motion token list length > 1:', name)
                                        raise Exception()
                                    m_token_len.append(len(m_token_new))
                                    m_token_dict_new[p_n] = m_token_new

                                # m_token_list_new = [tokens[int(f_tag*fps/unit_length) : int(to_tag*fps/unit_length)] for tokens in m_token_list if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length)]

                                # Debug
                                for m_len in m_token_len:
                                    assert m_len == m_token_len[0]

                                if m_token_len[0] == 0:
                                    continue

                                # if len(m_token_list_new) == 0:
                                #     continue

                                new_name = '%s_%f_%f'%(name, f_tag, to_tag)

                                data_dict[new_name] = {'m_token_dict': m_token_dict_new,
                                                       'text': [text_dict]}
                                new_name_list.append(new_name)

                        except Exception as e:
                            if self.print_warning:
                                print('Unable to process:', name)
                                print(e)

                if flag:
                    # data_dict[name] = {'m_token_list': m_token_list,
                    #                    'text': text_data}
                    data_dict[name] = {'m_token_dict': m_token_dict,
                                       'text': text_data}
                    new_name_list.append(name)

            except Exception as e:
                if self.print_warning:
                    print('Unable to load:', name, '. Usually because the motion was filtered in dataset_tokenize.py .')
                    print(e)

        self.data_dict = data_dict
        self.name_list = new_name_list

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        m_token_dict, text_list = data['m_token_dict'], data['text']


        # The length of m_token_list always equals to 1, thus it seems the random.choice has no effect.
        #  It just moves out the numpy.array from the list.
        # m_tokens = random.choice(m_token_list)
        # print(len(m_tokens))

        for p_n in self.parts_name:  # ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']
            assert len(m_token_dict[p_n]) == 1

        Root        = m_token_dict['Root'][0]
        R_Leg       = m_token_dict['R_Leg'][0]
        L_Leg       = m_token_dict['L_Leg'][0]
        Backbone    = m_token_dict['Backbone'][0]
        R_Arm       = m_token_dict['R_Arm'][0]
        L_Arm       = m_token_dict['L_Arm'][0]

        text_data = random.choice(text_list)
        caption = text_data['caption']

        # Choose to do random dropping
        coin = np.random.choice([False, False, True])
        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                Root = Root[:-1]
                R_Leg = R_Leg[:-1]
                L_Leg = L_Leg[:-1]
                Backbone = Backbone[:-1]
                R_Arm = R_Arm[:-1]
                L_Arm = L_Arm[:-1]
                # m_tokens = m_tokens[:-1]
            else:
                Root = Root[1:]
                R_Leg = R_Leg[1:]
                L_Leg = L_Leg[1:]
                Backbone = Backbone[1:]
                R_Arm = R_Arm[1:]
                L_Arm = L_Arm[1:]
                # m_tokens = m_tokens[1:]

        padded_tokens = []
        token_len = Root.shape[0]  # Debug, make sure all parts motions have the same token len.
        for m_tokens in [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]:
            # Getting the motion_token length
            m_tokens_len = m_tokens.shape[0]

            # Debug
            assert token_len == m_tokens_len

            # Add the end token, and decide if pad the motion_token.
            if m_tokens_len+1 < self.max_motion_length:
                m_t = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_length-1-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0)
            else:
                m_t = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)

            m_t = m_t.reshape(-1)
            padded_tokens.append(m_t)

        # unravel
        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = padded_tokens

        # caption: pure text caption
        # Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm: tokenized motion sequence
        # token_len: length, not include the End token
        return caption, Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm, token_len




def DATALoader(
        dataset_name, vq_dir, unit_length, codebook_size,
        batch_size, num_workers=8
):

    train_loader = torch.utils.data.DataLoader(
        Text2MotionDataset(
            dataset_name, vq_dir, unit_length, codebook_size,
        ),
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        #collate_fn=collate_fn,
        drop_last=True
    )
    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


