from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import _gather_feature, _tranpose_and_gather_feature, flip_tensor


def _nms(heat, kernel=3):
  hmax = F.max_pool2d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
  keep = (hmax == heat).float()       #每个元素互相比较 相同的留下.
  return heat * keep

def middle_nms(heat, kernel=3):
  hmax = F.max_pool2d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
  keep = (hmax == heat).float()  # 每个元素互相比较 相同的留下.
  one = torch.nonzero(keep)
  return one

def _topk(scores, K=40):
  # score size (batch, class, h, w)
  batch, cat, height, width = scores.size()

  topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)     # 这表示从一行中取前K个最大值 即每一张热图中去取;
  # topk_inds = topk_inds % (height * width)     # 无用
  topk_ys = (topk_inds // width).int().float()   # /
  topk_xs = (topk_inds % width).int().float()

  topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)       # 不区分类别的K个最大值
  topk_clses = (topk_ind // K).int()          # /
  topk_inds = _gather_feature(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_ys = _gather_feature(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_xs = _gather_feature(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

  return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(hmap, regs, w_h_, K=100, *args):
  if len(args) == 6:
    hmap, regs, w_h_ = args[0], args[1], args[2]

  batch, cat, height, width = hmap.shape
  hmap = torch.sigmoid(hmap)
  hmap[hmap < 0.3] = 0
  # print(hmap)
  # if flip test
  if batch > 1:
    hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
    w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
    regs = regs[0:1]

  batch = 1

  # print("hmap: ", hmap.size())
  # print("w_h_: ", w_h_.size())
  # print("regs: ", regs.size())

  hmap = _nms(hmap)  # perform nms on heatmaps

  scores, inds, clses, ys, xs = _topk(hmap, K=K)

  regs = _tranpose_and_gather_feature(regs, inds)
  regs = regs.view(batch, K, 2)
  xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
  ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

  w_h_ = _tranpose_and_gather_feature(w_h_, inds)
  w_h_ = w_h_.view(batch, K, 2)

  clses = clses.view(batch, K, 1).float()
  scores = scores.view(batch, K, 1)
  bboxes = torch.cat([xs - w_h_[..., 0:1] / 2,
                      ys - w_h_[..., 1:2] / 2,
                      xs + w_h_[..., 0:1] / 2,
                      ys + w_h_[..., 1:2] / 2], dim=2)
  detections = torch.cat([bboxes, scores, clses], dim=2)
  return detections


