import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.post_process import ctdet_decode

def _neg_loss_slow(preds, targets):
  pos_inds = targets == 1  # todo targets > 1-epsilon ?
  neg_inds = targets < 1  # todo targets < 1-epsilon ?

  neg_weights = torch.pow(1 - targets[neg_inds], 4)

  loss = 0
  for pred in preds:
    pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
      loss = loss - neg_loss
    else:
      loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def _neg_loss(preds, targets):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
      Arguments:
      preds (B x c x h x w)
      gt_regr (B x c x h x w)
  '''
  pos_inds = targets.eq(1).float()
  neg_inds = targets.lt(1).float()

  neg_weights = torch.pow(1 - targets, 4)

  loss = 0
  for pred in preds:
    pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
      loss = loss - neg_loss
    else:
      loss = loss - (pos_loss + neg_loss) / num_pos
  return loss / len(preds)

def _reg_loss(regs, gt_regs, mask):
  mask = mask[:, :, None].expand_as(gt_regs).float()
  loss = sum(F.l1_loss(r * mask, gt_regs * mask, reduction='sum') / (mask.sum() + 1e-4) for r in regs)
  return loss / len(regs)


def middle_hmap_loss(hmap, gt_hmap, yy_hmap, yy_gt_hmap):
  hmap_sum = hmap - yy_hmap
  gt_hmao_sum = gt_hmap - yy_gt_hmap
  hmap_ca = hmap_sum - gt_hmao_sum
  hmap_loss = sum(r for r in hmap_ca)
  return hmap_loss / len(hmap)

def middle_regs_loss(regs, gt_regs, reg_mask, yy_regs, yy_gt_regs, yy_reg_mask):
  reg_mask = reg_mask[:, :, None].expand_as(gt_regs).float()
  yy_reg_mask = yy_reg_mask[:, :, None].expand_as(yy_gt_regs).float()
  regs_sum_mask = (reg_mask - yy_reg_mask) / 2
  regs_sum = (regs - yy_regs) / 2
  regs_ca =regs_sum - regs_sum_mask
  regs_loss = sum(r for r in regs_ca)
  return regs_loss / len(regs)

def middle_w_h__loss(w_h_, gt_w_h_, w_h__mask, yy_w_h_, yy_gt_w_h_, yy_w_h__mask):
  w_h__mask = w_h__mask[:, :, None].expand_as(gt_w_h_).float()
  yy_w_h__mask = yy_w_h__mask[:, :, None].expand_as(yy_gt_w_h_).float()
  w_h__sum_mask = (w_h__mask - yy_w_h__mask) / 2
  w_h_sum = (w_h_ - yy_w_h_) / 2
  w_h__ca = w_h_sum - w_h__sum_mask
  w_h__loss = sum(r for r in w_h__ca)
  return w_h__loss / len(w_h_)

def middle_sum_loss(hmap_loss, regs_loss, w_h__loss):
  sum_loss = hmap_loss + regs_loss + w_h__loss
  return sum_loss
  # loss = sum(F.l1_loss(r * mask, gt_regs * mask, reduction='sum') / (mask.sum() + 1e-4) for r in regs)
  # return loss / len(regs)

def middle_loss(hmap, gt_hmap, reg, gt_reg, w_h_, gt_w_h_):
  cd = ctdet_decode(hmap, reg, w_h_)
  gt_cd = ctdet_decode(gt_hmap, gt_reg, gt_w_h_)
  return cd, gt_cd

def _similar_loss(f1, f2): # targets是用于判断图片中是否有阴影，没有的话，就不求解similar_loss。这个可以去掉
  """
  将用于计算两张特征图的相似度，使得两张特征图更为相近
  argus:
  f1 : (B, C, W, H)
  f2 : (B, C, W, H)
  """
  # torch.abs(torch.kl_div(f1, f2))   # 如果现在想用Y指导X，第一个参数要传X，第二个要传Y。
  # loss = 0
  # for i in range(f1.size(0)):
  #   loss += torch.log(F.mse_loss(f1[i], f2[i], reduction='sum')) / torch.log(F.mse_loss(f1, f2, reduction='sum'))
  loss = F.mse_loss(f1, f2, reduction='sum')

  # b = f1.size(0)
  # pos_ind = targets.eq(1).float()
  # pos_ind = pos_ind.view(b, -1).contiguous().sum(axis=1)
  #
  # f1 = f1.view(b, -1).contiguous()
  # f2 = f2.view(b, -1).contiguous()
  # pixel_num = f1.size(1)
  #
  # # mse loss
  # loss = torch.sum(torch.pow(torch.subtract(f1, f2), 2).sum(axis=1) * pos_ind / pixel_num) / b
  return loss

