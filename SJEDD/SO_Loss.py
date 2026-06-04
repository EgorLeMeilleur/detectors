import torch, pdb, copy
import torch.nn as nn
import numpy as np
import itertools as it

class pLoss_all_fidelity(nn.Module):
    def __init__(self, hexG, dataset='ffsc'):
        super(pLoss_all_fidelity, self).__init__()
        self.legal_state = hexG[0]
        self.label_state_idx = hexG[1]
        self.fideltiy_loss = cal_loss_multilabel_fideltiy(dataset=dataset)
        self.dataset = dataset

    def transfer_logtis(self, dataset, pMargin):
        if dataset in ['ffsc']:
            logits_l1 = pMargin[:, 0]
            logits_l2 = pMargin[:, 1:6]
            logits_l3 = pMargin[:, 6:]
        else: # for ff++
            logits_l1 = pMargin[:, 0]
            logits_l2 = pMargin[:, 1:4]
            logits_l3 = pMargin[:, 4:]
        return logits_l1, logits_l2, logits_l3

    def forward(self, f, y):
        y = y.float()
        pMargin = torch.zeros((f.shape[0], f.shape[1])).to(f.device)

        S = self.legal_state.to(f.device)
        # f: bzxn S: (n+1)xn
        potential = torch.mm(S.double(), f.T.double())
        max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
        J = torch.exp(potential - max_sf)  # (n+1)xbz
        # J = potential
        z_ = torch.sum(J, dim=0)

        id_num = _check_abnornal(z_)

        for i in range(f.shape[1]):
            pMargin[:, i] = torch.sum(J[S[:, i] > 0, :], dim=0) / z_

        logits_l1, logits_l2, logits_l3 = self.transfer_logtis(self.dataset, pMargin)
        total_loss, all_loss = self.fideltiy_loss(logits_l1, logits_l2, logits_l3, y)
        return all_loss, pMargin


    def infer(self, f):
        pMargin = torch.zeros((f.shape[0], f.shape[1])).to(f.device)
        S = self.legal_state.to(f.device)
        potential = torch.mm(S.double(), f.T.double())

        max_sf, _ = torch.max(potential, dim=0)  # to solve the overflow or underflow
        J = torch.exp(potential - max_sf)  # (n+1)xbz
        z_ = torch.sum(J, dim=0)

        # for Z_ is inf, ignore its pMargin loss
        id_num = _check_abnornal(z_)

        for i in range(f.shape[1]):
            pMargin[:, i] = torch.sum(J[S[:, i] > 0, :], dim=0) / z_

        return pMargin


class cal_loss_multilabel_fideltiy(torch.nn.Module):
    def __init__(self, loss_mask=False, dataset='ffsc'):
        super().__init__()
        self.dataset = dataset

        self.bce = Fidelity_Loss_binary()
        self.abstract_face_loss = Multi_Fidelity_Loss(loss_mask)
        self.local_face_loss = Multi_Fidelity_Loss(loss_mask)

    def forward(self, p1, p2, p3, target):
        g1, g2, g3 = transfer_label(target, dataset = self.dataset)

        loss1 = self.bce(p1, g1)
        loss2 = self.abstract_face_loss(p2, g2)
        loss3 = self.local_face_loss(p3, g3)

        total_loss = loss1 + loss2 + loss3
        all_loss = [loss1, loss2, loss3]
        return total_loss, all_loss

class Fidelity_Loss_binary(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.esp = 1e-8

    def forward(self, p, g):
        loss = 1 - (torch.sqrt(p * g + self.esp) + torch.sqrt((1 - p) * (1 - g) + self.esp))

        return torch.mean(loss)


class Multi_Fidelity_Loss(torch.nn.Module):
    def __init__(self, loss_mask=False):
        super(Multi_Fidelity_Loss, self).__init__()
        self.esp = 1e-8
        self.loss_mask = loss_mask

    def forward(self, p, g):

        if not self.loss_mask:
            return self.forward_wo_mask_matrix(p, g)
        else:
            return self.forward_w_mask(p, g)

    def forward_wo_mask_matrix(self, p, g):
        p_i = p.view(-1, p.size(1), 1)
        g_i = g.view(-1, g.size(1), 1)
        loss_i = 1 - (torch.sqrt(p_i * g_i + self.esp) + torch.sqrt((1 - p_i) * (1 - g_i) + self.esp))

        loss = torch.mean(loss_i, dim=1)
        mean_loss = torch.mean(loss)
        return mean_loss


    def forward_w_mask(self, p, g):
        indx = torch.where(g.sum(dim=1) == 0)[0]
        mask = torch.ones((g.size(0),))
        mask[indx]=0
        mask = mask.to(p.device)

        loss = 0
        for i in range(p.size(1)):
            p_i = p[:, i] * mask
            g_i = g[:, i] * mask
            g_i = g_i.view(-1, 1)
            p_i = p_i.view(-1, 1)
            loss_i = 1 - (torch.sqrt(p_i * g_i + self.esp) + torch.sqrt((1 - p_i) * (1 - g_i) + self.esp))
            loss = loss + loss_i
        loss = loss / p.size(1)

        return torch.mean(loss)


# utils for pLoss
def _check_abnornal(z_):
    if np.inf in z_:
        pdb.set_trace()
        idx = z_ == np.inf
        id_num = [i for i, v in enumerate(list(idx)) if v == True]
    else:
        id_num = [-1]
    return id_num

def transfer_label(target, dataset=''):
    if dataset in ['ffpp']:
        gt_level_1 = target[:, 0]
        gt_level_2 = target[:, 1: 4]
        gt_level_3 = target[:, 4:]
    else:
        gt_level_1 = target[:, 0]
        gt_level_2 = target[:, 1: 6]
        gt_level_3 = target[:, 6:]

    return gt_level_1, gt_level_2, gt_level_3