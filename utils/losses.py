import torch
import torch.nn as nn

class ReConsLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints):
        super(ReConsLoss, self).__init__()
        
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()
        
        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.nb_joints = nb_joints
        self.motion_dim = (nb_joints - 1) * 12 + 4 + 3 + 4
        
    def forward(self, motion_pred, motion_gt) :

        assert self.motion_dim == motion_pred.shape[-1] == motion_gt.shape[-1]

        loss = self.Loss(motion_pred[..., : self.motion_dim], motion_gt[..., :self.motion_dim])
        return loss
    
    def forward_vel(self, motion_pred, motion_gt) : 
        loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4], motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4])
        return loss
    

class ReConsLossBodyPart(nn.Module):

    def __init__(self, recons_loss, nb_joints):
        super(ReConsLossBodyPart, self).__init__()

        if recons_loss == 'l1':
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2':
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth':
            self.Loss = torch.nn.SmoothL1Loss()

        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.nb_joints = nb_joints
        # self.motion_dim = (nb_joints - 1) * 12 + 4 + 3 + 4

        if nb_joints == 22:  # t2m dataset (i.e. HumanML3D dataset)
            # Corresponding to [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]
            self.parts_motion_dim = [7, 50, 50, 60, 60, 60]

        elif nb_joints == 21:  # kit dataset
            self.parts_motion_dim = [7, 62, 62, 48, 48, 48]

        else:
            raise Exception()

    def forward(self, motion_pred_list, motion_gt_list):

        assert len(motion_pred_list) == len(motion_gt_list)

        loss_list = []
        for i in range(len(motion_gt_list)):
            motion_pred = motion_pred_list[i]
            motion_gt = motion_gt_list[i]
            motion_dim = self.parts_motion_dim[i]

            # The dimension should be the same.
            #  If so, there is no need to slice the motion_pred and motion_gt in the Loss function.
            assert motion_dim == motion_pred.shape[-1] == motion_gt.shape[-1]
            # todo: check if the slice of motion is necessary.
            loss = self.Loss(motion_pred[..., : motion_dim], motion_gt[..., :motion_dim])
            loss_list.append(loss)
        return loss_list

    def forward_vel(self, motion_pred_list, motion_gt_list):

        assert len(motion_pred_list) == len(motion_gt_list)

        loss_list = []
        for i in range(len(motion_gt_list)):
            motion_pred = motion_pred_list[i]  # (bs, nframes, vel_dim)
            motion_gt = motion_gt_list[i]  # # (bs, nframes, vel_dim)
            loss = self.Loss(motion_pred, motion_gt)
            loss_list.append(loss)

        return loss_list


class ReConsLossUpLow(nn.Module):

    def __init__(self, recons_loss, nb_joints):
        super(ReConsLossUpLow, self).__init__()

        if recons_loss == 'l1':
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2':
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth':
            self.Loss = torch.nn.SmoothL1Loss()

        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.nb_joints = nb_joints
        # self.motion_dim = (nb_joints - 1) * 12 + 4 + 3 + 4

        if nb_joints == 22:  # t2m dataset (i.e. HumanML3D dataset)
            # Corresponding to ['Upper_body', 'Lower_body']
            self.parts_motion_dim = [163, 167]

        elif nb_joints == 21:  # kit dataset
            raise NotImplementedError()
            # self.parts_motion_dim = [7, 62, 62, 48, 48, 48]

        else:
            raise Exception()

    def forward(self, motion_pred_list, motion_gt_list):

        assert len(motion_pred_list) == len(motion_gt_list)

        loss_list = []
        for i in range(len(motion_gt_list)):
            motion_pred = motion_pred_list[i]
            motion_gt = motion_gt_list[i]
            motion_dim = self.parts_motion_dim[i]

            # The dimension should be the same.
            #  If so, there is no need to slice the motion_pred and motion_gt in the Loss function.
            assert motion_dim == motion_pred.shape[-1] == motion_gt.shape[-1]
            # todo: check if the slice of motion is necessary.
            loss = self.Loss(motion_pred[..., : motion_dim], motion_gt[..., :motion_dim])
            loss_list.append(loss)
        return loss_list

    def forward_vel(self, motion_pred_list, motion_gt_list):

        assert len(motion_pred_list) == len(motion_gt_list)

        loss_list = []
        for i in range(len(motion_gt_list)):
            motion_pred = motion_pred_list[i]  # (bs, nframes, vel_dim)
            motion_gt = motion_gt_list[i]  # # (bs, nframes, vel_dim)
            loss = self.Loss(motion_pred, motion_gt)
            loss_list.append(loss)

        return loss_list



def gather_loss_list(loss_list):
    assert isinstance(loss_list, list)
    loss_sum = 0
    for i, loss in enumerate(loss_list):
        if i == 0:
            loss_sum = loss
        else:
            loss_sum = loss_sum + loss
    return loss_sum



