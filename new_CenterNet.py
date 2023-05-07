import numpy as np
import torch
import torch.nn as nn
import torchvision
from utils.post_process import ctdet_decode


class convolution(nn.Module):     # 普通卷积模块包含conv，bn relu
  def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
    super(convolution, self).__init__()
    pad = (k - 1) // 2
    self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
    self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
    self.relu = nn.ReLU(inplace=True)   # 当inplace=True的时候，会改变输入数据；当inplace=False的时候，不会改变输入数据

  def forward(self, x):
    conv = self.conv(x)
    bn = self.bn(conv)
    relu = self.relu(bn)
    return relu


class residual(nn.Module):  # 残差模块包含：首先x分为别进入两条通道，①conv1，bn1，relu1，conv2，bn2，②skip(x),最后相加再relu
  def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
    super(residual, self).__init__()
# inp_dim是输入的通道数, out_dim是渴望输出的通道数
    self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)  #(3,3)应该是卷积核的尺寸
    self.bn1 = nn.BatchNorm2d(out_dim)
    self.relu1 = nn.ReLU(inplace=True)   # 当inplace=True的时候，会改变输入数据；当inplace=False的时候，不会改变输入数据

    self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
    self.bn2 = nn.BatchNorm2d(out_dim)
# nn.Sequential()打包一套各种需要层的规格，也就是制造一种模板层供后面调用
    self.skip = nn.Sequential(nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
                              nn.BatchNorm2d(out_dim)) \
      if stride != 1 or inp_dim != out_dim else nn.Sequential()   # \是续行符
    self.relu = nn.ReLU(inplace=True)   # 如果x最后的通道数变了则skip执行上面的过程conv，bn，没变则执行下面的过程relu

  def forward(self, x):  # 组合各层，统一计算
    conv1 = self.conv1(x)
    bn1 = self.bn1(conv1)
    relu1 = self.relu1(bn1)

    conv2 = self.conv2(relu1)
    bn2 = self.bn2(conv2)

    skip = self.skip(x)
    return self.relu(bn2 + skip)

# 自适应融合模块
class Adaptive_fusion_module(nn.Module):
  def __init__(self, k, inp_dim_object, inp_dim_shadow,out_dim, stride=1, with_bn=True):
    super(Adaptive_fusion_module, self).__init__()
    self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride),
                               bias=False)  # (3,3)应该是卷积核的尺寸
    self.bn1 = nn.BatchNorm2d(out_dim)
    self.relu1 = nn.ReLU(inplace=True)  # 当inplace=True的时候，会改变输入数据；当inplace=False的时候，不会改变输入数据

    self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
    self.bn2 = nn.BatchNorm2d(out_dim)
    self.relu = nn.ReLU(inplace=True)  # 如果x最后的通道数变了则skip执行上面的过程conv，bn，没变则执行下面的过程relu

  def forward(self, x):  # 组合各层，统一计算
    conv1 = self.conv1(x)
    bn1 = self.bn1(conv1)
    relu1 = self.relu1(bn1)

    conv2 = self.conv2(relu1)
    bn2 = self.bn2(conv2)

    skip = self.skip(x)
    return self.relu(bn2 + skip)


# inp_dim -> out_dim -> ... -> out_dim
# make_layer 将空间分辨率降维
def make_layer(kernel_size, inp_dim, out_dim, modules, layer, stride=1):  # 这里的layer是指你要输入什么样的一种网络（例如residual残差）的大类（其中包含conv，bn，Relu，skip等等），传入的参数为类里面所拥有的变量
  layers = [layer(kernel_size, inp_dim, out_dim, stride=stride)]  # 这里把stride写死为1，起到降维作用
  layers += [layer(kernel_size, out_dim, out_dim) for _ in range(modules - 1)]
  return nn.Sequential(*layers)  # 返回一个由各种模块组成的列表


# inp_dim -> inp_dim -> ... -> inp_dim -> out_dim
# make_layer_revr 函数进行升维
def make_layer_revr(kernel_size, inp_dim, out_dim, modules, layer):
  layers = [layer(kernel_size, inp_dim, inp_dim) for _ in range(modules - 1)]  # 这里没有限制stride的大小，因此可以升维
  layers.append(layer(kernel_size, inp_dim, out_dim))
  return nn.Sequential(*layers)  # 返回一个由各种模块组成的列表


# key point layer
def make_kp_layer(cnv_dim, curr_dim, out_dim):  # 制造由不同卷积块所构成的层
  return nn.Sequential(convolution(3, cnv_dim, curr_dim, with_bn=False),  # 3为kernel_size(卷积尺寸)，cnv_dim为输入通道数，curr_dim，out_dim为输出通道数
                       nn.Conv2d(curr_dim, out_dim, (1, 1)))  # (1，1)为kernel_size(卷积尺寸)
# 其中有两个主要的函数 make_layer 和 make_layer_revr，其中make_layer 将空间分辨率降维，make_layer_revr 函数进行升维，所以将这个结构命名为 hourglass(沙漏)。
# 核心构建是一个递归函数，递归层数是通过 n 来控制，称之为 n 阶 hourglass 模块


class kp_module(nn.Module):         # kp是hourglass的子模块
  def __init__(self, n, dims, modules):
    super(kp_module, self).__init__()

    self.n = n  # n = 5

    curr_modules = modules[0]   # 2
    next_modules = modules[1]   # 2

    curr_dim = dims[0]  # 256
    next_dim = dims[1]  # 256

    # curr_mod x residual，curr_dim -> curr_dim -> ... -> curr_dim
    self.top = make_layer(3, curr_dim, curr_dim, curr_modules, layer=residual)  # curr_dim=256
    self.down = nn.Sequential()
    # curr_mod x residual，curr_dim -> next_dim -> ... -> next_dim
    self.low1 = make_layer(3, curr_dim, next_dim, curr_modules, layer=residual, stride=2)  # curr_dim=256 next_dim=256 curr_modules=2
    # next_mod x residual，next_dim -> next_dim -> ... -> next_dim
    if self.n > 1:
      self.low2 = kp_module(n - 1, dims[1:], modules[1:])  # 每次执行一次循环，n就会减一，以此来堆积模块low2，一共构建5个low2
    else:
      self.low2 = make_layer(3, next_dim, next_dim, next_modules, layer=residual)
    # curr_mod x residual，next_dim -> next_dim -> ... -> next_dim -> curr_dim
    self.low3 = make_layer_revr(3, next_dim, curr_dim, curr_modules, layer=residual)
    self.up = nn.Upsample(scale_factor=2)

  def forward(self, x):
    up1 = self.top(x)
    down = self.down(x)
    low1 = self.low1(down)
    low2 = self.low2(low1)
    low3 = self.low3(low2)
    up2 = self.up(low3)
    return up1 + up2


class exkp(nn.Module):   # 搭建hourglass网络结构，其中包含kp_module, residual, 等模块
  def __init__(self, n, nstack, dims, modules, cnv_dim=256, num_classes=8):    # num_classes更改预测的类别数，即最终热图输出的通道数
    super(exkp, self).__init__()

    self.nstack = nstack      # stack块的个数
    self.num_classes = num_classes  # 类别数，决定最后热图的输出通道

    curr_dim = dims[0]  # dim[0] == 256

    self.pre = nn.Sequential(convolution(7, 3, 128, stride=2),
                             residual(3, 128, curr_dim, stride=2))   # 输出通道为curr_dim，即dims[0]=256  3-128-256

    self.kps = nn.ModuleList([kp_module(n, dims, modules) for _ in range(nstack)])   # 输出通道为dims, 输出通道为dims

    self.cnvs = nn.ModuleList([convolution(3, curr_dim, cnv_dim) for _ in range(nstack)])  # 输出通道为curr_dim=256, 输出通道为cnv_dim

    self.cnv_sum_image = convolution(3, 512, 256)   # 给拼接后的特征图所做的卷积conv，bn，relu

    self.inters = nn.ModuleList([residual(3, curr_dim, curr_dim) for _ in range(nstack - 1)])  # 输出通道为curr_dim=256, 输出通道为curr_dim

    self.inters_ = nn.ModuleList([nn.Sequential(nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                                                nn.BatchNorm2d(curr_dim))
                                  for _ in range(nstack - 1)])      # 输出通道为curr_dim=256, 输出通道为curr_dim=256
    self.cnvs_ = nn.ModuleList([nn.Sequential(nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                                              nn.BatchNorm2d(curr_dim))
                                for _ in range(nstack - 1)])  # 输出通道为cnv_dim=256(得根具实际传入的参数来看), 输出通道为curr_dim=256
    # heatmap layers
    self.hmap = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, num_classes) for _ in range(nstack)])  # 预测热图,里面的三个参数都代表通道数，最终输出为num_classes.前两个只是中间过渡的通道数
    for hmap in self.hmap:
      hmap[-1].bias.data.fill_(-2.19)

    # regression layers
    self.regs = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])  # ；regs 是偏置量，预测偏移量
    self.w_h_ = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])  # ：w_h_是长宽， 预测长宽

    self.hmap1 = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, num_classes) for _ in range(nstack)])  # 预测热图,里面的三个参数都代表通道数，最终输出为num_classes.前两个只是中间过渡的通道数
    for hmap1 in self.hmap1:
      hmap1[-1].bias.data.fill_(-2.19)

    self.regs1 = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])  # ；regs 是偏置量，预测偏移量
    self.w_h_1 = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)])  # ：w_h_是长宽， 预测长宽

    self.relu = nn.ReLU(inplace=True)

  def forward(self, image):
    inter = self.pre(image)  # conv, residual  得到一张特征图inter

    for ind in range(self.nstack):  # small_hourglass的nstack为1，因此只会出现索引0,表示创建几个hourglass子模块
      kp = self.kps[ind](inter)  # 将inter(conv, residual)这种模块传进去，以构建hourglass子模块
      cnv = self.cnvs[ind](kp)    # 将kp的值传入cnvs中(nstack个conv，bn relu，即普通卷积块所堆积构成的模块)

    # 如果为small_hourglass则nstack为1，则不会执行下列if语句
      if ind < self.nstack - 1:   # ind=0,nstack-1=0, 即最后一次循环不执行下列语句，下列语句的作用是保存中间监督
        inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)  # inters_(conv, bn), cnvs_(conv, bn)
        inter = self.relu(inter)  # 再经历一个relu
        inter = self.inters[ind](inter)  # 最后经过residual

    hmap1 = self.hmap1[ind](cnv)
    regs1 = self.regs1[ind](cnv)
    w_h_1 = self.w_h_1[ind](cnv)
    out = []
    out.append([hmap1, regs1, w_h_1])
    # print("hmap1: ", hmap1.size())
    # print("regs: ", regs.size())
    # print("w_h_: ", w_h_.size())

    bbox = []
    cnv2 = []
    # print("image.shape: ", image.shape[0])
    batch_size = image.shape[0]
    for i in range(batch_size):
      hmap1_s = hmap1[i:i+1]
      regs1_s = regs1[i:i+1]
      w_h_1_s = w_h_1[i:i+1]
      # print("hmap1: ", hmap1.size())
      # print("regs1: ", regs1.size())
      # print("w_h_1: ", w_h_1.size())

      cd = ctdet_decode(hmap1_s, regs1_s, w_h_1_s)
      bbox.append(cd)  # 找出切割的位置(x,y,x,y)
      # print("bbox_all: ", bbox)


      bbox_w1 = bbox[i][0][0][2] - bbox[i][0][0][0]  # 求出切割框的宽
      bbox_w2 = 1.5 * bbox_w1  # 求出切割框1.5倍的宽
      Increment = bbox_w2 - bbox_w1  # 求出切割框宽的增加量
      Increment_half = Increment / 2  # 求出切割框宽的增加量的二分之一

      high = bbox[i][0][0][1]
      bbox[i][0][0][3] = high  # 将y2限制在目标物体框的上方
      bbox[i][0][0][1] = 0     # 将长h变成特征图图的长度，若注释掉，则取目标物以下的特征区域

      bbox[i][0][0][0] = bbox[i][0][0][0] - Increment_half  # 将x1向左偏移
      bbox[i][0][0][2] = bbox[i][0][0][2] + Increment_half  # 将x2向右偏移

      bbox_list1 = []
      for n in range(4):  # 将x1,y1,x2,y2依次添加进bbox_list1列表里
        bbox_list1.append(bbox[i][0][0][n])

      device = cnv.device
      bbox_list1 = torch.Tensor(bbox_list1).unsqueeze(0).to(device)  # list转换成tensor并且升维度

      # bbox_list1 = torch.Tensor(bbox_list1).unsqueeze(0)  # list转换成tensor并且升维度

      bbox_list2 = []
      bbox_list2.append(bbox_list1)   # 将bbox_list1这个列表添加进bbox_list2这个列表里。

      # print("bbox_list1: ", bbox_list1)
      # print("bbox_list2: ", bbox_list2)
      # print("type: ", bbox_list)

      roi_image = torchvision.ops.roi_align(input=cnv[i:i+1], boxes=bbox_list2, output_size=(cnv.size(2), cnv.size(3)))   # 按bbox的坐标进行切割
      # print("cnv[i:i+1]: ", cnv[i:i+1].size())
      # print("roi_image: ", roi_image.size())

      # print("cnv[i]: ", cnv[i].size())
      # print("roi_image[0]: ", roi_image[0].size())


      cat  = torch.cat((cnv[i], roi_image[0]), 0)
      # print("cat: ", cat.size())
      cnv2.append(cat)  # 将切割结果与backbone最后一张特征图拼接cat
      # cnv2.append(cnv[i] + roi_image[0])  # 将切割结果与backbone最后一张特征图相加
      # print("cnv2.size(): ", np.array(cnv2).shape)

    cnv3 = torch.stack(cnv2)
    # print("cnv3.size(): ", cnv3.size())

    sum_image = self.cnv_sum_image(cnv3)
    # print("sum_image.size(): ", sum_image.size())

    outs = []
    for ind in range(self.nstack):
      if self.training or ind == self.nstack - 1:
        outs.append([self.hmap[ind](sum_image), self.regs[ind](cnv), self.w_h_[ind](cnv)])  # 输出三种预测，热图，偏移量，长宽

    if self.training:
      return [outs[0] + out[0]]  # 返回的是一个存放了张量的列表
    else:
      return outs

# 反斜杠是续行符
get_hourglass = \
  {'large_hourglass':
     exkp(n=5, nstack=2, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4]),
   'small_hourglass':
     exkp(n=5, nstack=1, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4])}  # dim为通道数，modules为叠加的块

if __name__ == '__main__':
  from collections import OrderedDict
  from utils.utils import count_parameters, count_flops, load_model


  def hook(self, input, output):
    print(output.data.cpu().numpy().shape)
    # pass


  net = get_hourglass['small_hourglass']     # 177行 这里更改所使用的模型配置
  # load_model(net, '../ckpt/pretrain/checkpoint.t7')
  count_parameters(net)
  count_flops(net, input_size=512)

  # for m in net.modules():
  #   if isinstance(m, nn.Conv2d):
  #     m.register_forward_hook(hook)
  #
  # with torch.no_grad():
  #   y = net(torch.randn(2, 3, 512, 512).cuda())
  # print(y.size())
