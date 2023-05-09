import random
import sys
import time
import colorsys

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
# from data import BaseTransform
# from data import VOC_CLASSES as labelmap
# from models.refinedet import build_refinedet
from nets.cnv.hourglass_all_addloss import get_hourglass  # 改成CenterNet的
from utils.utils import load_model  # CenterNet加载模型权重的函数
import argparse
import numpy as np
import cv2 as cv
import os
from PIL import Image
from PyQt5.QtCore import QBasicTimer

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

import torch.utils.data

from datasets.coco import COCO_MEAN, COCO_STD, COCO_NAMES
from datasets.pascal import VOC_MEAN, VOC_STD, VOC_NAMES

from utils.image import transform_preds, get_affine_transform
from utils.post_process import ctdet_decode

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib
matplotlib.use('TkAgg')

COCO_COLORS = sns.color_palette('hls', len(COCO_NAMES))
VOC_COLORS = sns.color_palette('hls', len(VOC_NAMES))


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


global SORT, chose_dir, img_names, img_cv
img_names = None

class MainUi(QtWidgets.QMainWindow, QWidget):
    def __init__(self):
        super().__init__()

        global img_cv
        img_cv = None
        # 模型初始化
        self.net = None
        self.init_ui()

    def init_ui(self):
        self.setFixedSize(1600, 900)
        self.main_widget = QtWidgets.QWidget()  # 创建窗口主部件
        self.main_layout = QtWidgets.QGridLayout()  # 创建主部件的网格布局
        self.main_widget.setLayout(self.main_layout)  # 设置窗口主部件布局为网格布局

        self.left_widget = QtWidgets.QWidget()  # 创建左侧部件
        self.left_widget.setObjectName('left_widget')
        self.left_layout = QtWidgets.QGridLayout()  # 创建左侧部件的网格布局层
        self.left_widget.setLayout(self.left_layout)  # 设置左侧部件布局为网格

        self.right_widget = QtWidgets.QWidget()  # 创建右侧部件
        self.right_widget.setObjectName('right_widget')
        self.right_layout = QtWidgets.QGridLayout()  # 创建右侧部件的网格布局层
        self.right_widget.setLayout(self.right_layout)  # 设置右侧部件布局为网格

        self.main_layout.addWidget(self.left_widget, 0, 0, 12, 1)  # 左侧部件在第0行第0列，占8行3列
        self.main_layout.addWidget(self.right_widget, 0, 2, 12, 10)  # 右侧部件在第0行第3列，占8行9列

        self.setCentralWidget(self.main_widget)  # 设置窗口主部件
        self.main_widget.setStyleSheet('''QWidget{border-radius:7px;background-color:white;}''')


        self.left_button_1 = QtWidgets.QPushButton( "打开文件夹")
        self.left_button_1.setObjectName('left_button')
        c = self.left_button_1
        c.clicked.connect(self.openimage)

        self.left_button_2 = QtWidgets.QPushButton("上一张")
        self.left_button_2.setObjectName('left_button')
        b = self.left_button_2
        b.clicked.connect(self.backimage)

        self.left_button_3 = QtWidgets.QPushButton("下一张")
        self.left_button_3.setObjectName('left_button')
        a = self.left_button_3
        a.clicked.connect(self.nextimage)

        self.left_button_4 = QtWidgets.QPushButton("图像另存")
        self.left_button_4.setObjectName('left_button')
        e=self.left_button_4
        e.clicked.connect(self.saveimage)

        self.left_button_5 = QtWidgets.QPushButton("图像检测")
        self.left_button_5.setObjectName('left_button')
        f = self.left_button_5
        f.clicked.connect(self.detectimage)

        self.left_button5_5 = QtWidgets.QPushButton("加载模型权重")
        self.left_button5_5.setObjectName('left_button')
        n = self.left_button5_5
        n.clicked.connect(self.start_image2)
        n.clicked.connect(self.load_model)

        self.left_xxx = QtWidgets.QPushButton(" ")

        self.left_layout.addWidget(self.left_button_1, 0, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_2, 1, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_3, 2, 0, 1, 3)

        self.left_layout.addWidget(self.left_button_4, 3, 0, 1, 3)
        self.left_layout.addWidget(self.left_button5_5, 4, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_5, 5, 0, 1, 3)
        # self.left_layout.addWidget(self.left_button6_5, 9, 0, 1, 3)
        self.label = QLabel(self)
        self.label.setFixedSize(300, 300)
        self.label.move(160, 460)

        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(280, 800, 1000, 25)
        self.timer = QBasicTimer()
        self.step = 0
        self.pbar.setStyleSheet(
            "QProgressBar {border: 2px solid grey; border-radius: 5px;   background-color: white;}QProgressBar::chunk {   background-color: grey;   width: 10px;}QProgressBar {   border: 2px solid grey;   border-radius: 5px;   text-align: center;}")

        self.label2 = QLabel(self)
        self.label2.setFixedSize(600, 600)
        self.label2.move(480, 160)

        self.label4 = QLabel(self)
        self.label4.setFixedSize(420, 630)
        self.label4.move(1100, 130)

        # 调整右侧表层深灰色区域的大小及偏移位置
        self.label5 = QLabel(self)
        self.label5.setFixedSize(420, 600)
        self.label5.move(1100, 160)

        self.label6 = QLabel(self)
        self.label6.setFixedSize(300, 270)
        self.label6.move(160, 160)

        self.label2.setStyleSheet("QLabel{background-color:rgb(74,74,74);}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:黑体;}"
                                 )
        self.label.setStyleSheet("QLabel{background-color:rgb(74,74,74);}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:黑体;}"
                                 )
        # 修改右侧框上下部分颜色
        self.label4.setStyleSheet("QLabel{background-color:grey;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:黑体;}"
                                 )
        self.label5.setStyleSheet("QLabel{background-color:rgb(74,74,74);}"
                                  "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:黑体;}"
                                  )
        self.label6.setStyleSheet("QLabel{background-color:rgb(74,74,74);}"
                                  "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:黑体;}"
                                  )

        # 更改左侧字体颜色
        self.left_widget.setStyleSheet('''
            QPushButton{border:none;color:black;}
            QPushButton#left_label{
                border:none;
                border-bottom:1px solid white;
                font-size:18px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }
            QPushButton#left_button:hover{border-left:20px solid red;font-weight:700;}
        ''')
        self.setWindowTitle("基于CenterNet的声呐图像检测系统")
        self.setWindowIcon(QIcon(r'E:\sonar\CenterNet\广海校徽.jpeg'))

        self.setWindowOpacity(10)  # 设置窗口透明度
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置窗口背景透明

        self.wenben = QLabel(self)
        self.wenben.setText(u"基于CenterNet的声呐图像检测系统")
        self.wenben.setGeometry(510, 40, 600, 60)
        self.wenben.setStyleSheet("QLabel{background:white;}"
                                  "QLabel{color:black;font-size:35px;font-weight:white;font-family:黑体;}"
                                  )
        self.wenben1 = QLabel(self)
        self.wenben1.setText(u"检测输入")
        self.wenben1.setGeometry(160, 430, 300, 30)
        self.wenben1.setStyleSheet("QLabel{background:grey;}"
                                  "QLabel{color:white;font-size:20px;font-weight:white;font-family:黑体;}"
                                  )
        self.wenben2 = QLabel(self)
        self.wenben2.setText(u"检测结果")
        self.wenben2.setGeometry(480, 130, 600, 30)
        self.wenben2.setStyleSheet("QLabel{background:grey;}"
                                  "QLabel{color:white;font-size:20px;font-weight:white;font-family:黑体;}"
                                  )
        self.wenben3 = QLabel(self)
        self.wenben3.setText(u"基本信息")
        self.wenben3.setGeometry(160, 130, 300, 30)
        self.wenben3.setStyleSheet("QLabel{background:grey;}"
                                   "QLabel{color:white;font-size:20px;font-weight:white;font-family:黑体;}"
                                   )
        self.wenben4 = QLabel(self)
        self.wenben4.setText(u"声呐图像检测结果")
        self.wenben4.setGeometry(1105, 130, 410, 30)
        self.wenben4.setStyleSheet("QLabel{background-color:grey;}"
                                   "QLabel{color:white;font-size:20px;font-weight:white;font-family:黑体;}"
                                   )
        self.wenben5 = QLabel(self)
        self.wenben6 = QLabel(self)


    def timerEvent(self, e):

        if self.step >= 100:
            self.step = 0
            self.pbar.setValue(self.step)
            self.timer.stop()
            return
        self.step = self.step + 1
        self.pbar.setValue(self.step)

    def doAction(self, value):
        print("do action")
        if self.timer.isActive():
            self.timer.stop()

        else:
            self.timer.start(100, self)

    # 打开文件夹
    def openimage(self):
        global SORT, chose_dir, img_names, img_cv
        SORT = 0

        chose = QFileDialog.getExistingDirectory(self, '选取文件夹', os.getcwd())
        print('选取的文件夹为：')
        print(chose)
        print('\n')
        if not chose:
            pass
        else:
            chose_dir = chose
            img_names = os.listdir(chose_dir)
            print(img_names)
            print(len(img_names))

            img_path = os.path.join(chose_dir, img_names[SORT])
            print('正在显示的图片：')
            print(img_path)
            print('\n')

            img_cv = cv.imread(img_path)
            img_rgb = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)
            img = QtGui.QImage(img_rgb, img_rgb.shape[1], img_rgb.shape[0], QtGui.QImage.Format_RGB888)

            img = QtGui.QPixmap(img).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(img)

    # 上一张图片
    def backimage(self):
        self.label.setPixmap(QPixmap(""))
        self.label2.setPixmap(QPixmap(""))
        self.wenben5.setText('')
        self.wenben6.setText('')

        global SORT, chose_dir, img_names, img_cv
        SORT -= 1
        if SORT < 0:
            SORT = len(img_names)
        img_path = os.path.join(chose_dir, img_names[SORT])
        print('正在显示的图片：')
        print(img_path)
        print('\n')

        img_cv = cv.imread(img_path)
        img_rgb = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)
        img = QtGui.QImage(img_rgb, img_rgb.shape[1], img_rgb.shape[0], QtGui.QImage.Format_RGB888)

        img = QtGui.QPixmap(img).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(img)

    # 下一张图片
    def nextimage(self):
        self.label.setPixmap(QPixmap(""))
        self.label2.setPixmap(QPixmap(""))
        self.wenben5.setText('')
        self.wenben6.setText('')

        global SORT, chose_dir, img_names, img_cv
        SORT += 1
        if SORT >= len(img_names):
            SORT = 0
        img_path = os.path.join(chose_dir, img_names[SORT])
        print('正在显示的图片：')
        print(img_path)
        print('\n')

        img_cv = cv.imread(img_path)
        img_rgb = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)
        img = QtGui.QImage(img_rgb, img_rgb.shape[1], img_rgb.shape[0], QtGui.QImage.Format_RGB888)

        img = QtGui.QPixmap(img).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(img)

    # 保存图片
    def saveimage(self):
        global SORT, chose_dir, img_names, img_cv
        if not img_names:
            pass
        else:
            save_path, _ = QFileDialog.getSaveFileName(self, '保存图片', img_names[SORT])
            cv.imwrite(save_path, img_cv)
            print('已保存的图片：')
            print(save_path)
            print('\n')

    # 模型预测图片
    def detectimage(self):
        global img_cv, img_names
        if not img_names:
            pass
        else:
            if not self.net:
                pass
            else:
                cfg.device = torch.device('cuda')
                torch.backends.cudnn.benchmark = False

                max_per_image = 100
                img_path = os.path.join(chose_dir, img_names[SORT])
                image = cv.imread(img_path)
                height, width = image.shape[0:2]
                padding = 127 if 'hourglass' in cfg.arch else 31
                imgs = {}
                for scale in cfg.test_scales:
                    new_height = int(height * scale)
                    new_width = int(width * scale)

                    if cfg.img_size > 0:
                        img_height, img_width = cfg.img_size, cfg.img_size
                        center = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
                        scaled_size = max(height, width) * 1.0
                        scaled_size = np.array([scaled_size, scaled_size], dtype=np.float32)
                    else:
                        img_height = (new_height | padding) + 1
                        img_width = (new_width | padding) + 1
                        center = np.array([new_width // 2, new_height // 2], dtype=np.float32)
                        scaled_size = np.array([img_width, img_height], dtype=np.float32)

                    img = cv.resize(image, (new_width, new_height))
                    trans_img = get_affine_transform(center, scaled_size, 0, [img_width, img_height])
                    img = cv.warpAffine(img, trans_img, (img_width, img_height), flags=cv.INTER_NEAREST)

                    img = img.astype(np.float32) / 255.
                    img -= np.array(COCO_MEAN if cfg.dataset == 'coco' else VOC_MEAN, dtype=np.float32)[None, None, :]
                    img /= np.array(COCO_STD if cfg.dataset == 'coco' else VOC_STD, dtype=np.float32)[None, None, :]
                    img = img.transpose(2, 0, 1)[None, :, :, :]  # from [H, W, C] to [1, C, H, W]

                    if cfg.test_flip:
                        img = np.concatenate((img, img[:, :, :, ::-1].copy()), axis=0)  # 水平拼接

                    imgs[scale] = {'image': torch.from_numpy(img).float(),
                                   'center': np.array(center),
                                   'scale': np.array(scaled_size),
                                   'fmap_h': np.array(img_height // 4),
                                   'fmap_w': np.array(img_width // 4)}

                reports = []  # 创建列表，保存输出结果
                with torch.no_grad():
                    detections = []
                    for scale in imgs:
                        imgs[scale]['image'] = imgs[scale]['image'].to(cfg.device)

                        output = self.net(imgs[scale]['image'])[-1]
                        dets = ctdet_decode(*output, K=cfg.test_topk)
                        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]

                        top_preds = {}
                        dets[:, :2] = transform_preds(dets[:, 0:2],
                                                      imgs[scale]['center'],
                                                      imgs[scale]['scale'],
                                                      (imgs[scale]['fmap_w'], imgs[scale]['fmap_h']))
                        dets[:, 2:4] = transform_preds(dets[:, 2:4],
                                                       imgs[scale]['center'],
                                                       imgs[scale]['scale'],
                                                       (imgs[scale]['fmap_w'], imgs[scale]['fmap_h']))
                        cls = dets[:, -1]
                        for j in range(80):
                            inds = (cls == j)
                            top_preds[j + 1] = dets[inds, :5].astype(np.float32)
                            top_preds[j + 1][:, :4] /= scale

                        detections.append(top_preds)

                    bbox_and_scores = {}
                    for j in range(1, 81 if cfg.dataset == 'coco' else 21):
                        bbox_and_scores[j] = np.concatenate([d[j] for d in detections], axis=0)
                        # if len(cfg.test_scales) > 1:
                        #   soft_nms(bbox_and_scores[j], Nt=0.5, method=2)
                    scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1, 81 if cfg.dataset == 'coco' else 21)])

                    if len(scores) > max_per_image:
                        kth = len(scores) - max_per_image
                        thresh = np.partition(scores, kth)[kth]
                        for j in range(1, 81 if cfg.dataset == 'coco' else 21):
                            keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
                            bbox_and_scores[j] = bbox_and_scores[j][keep_inds]


                    fig = plt.figure(0)
                    colors = COCO_COLORS if cfg.dataset == 'coco' else VOC_COLORS
                    names = COCO_NAMES if cfg.dataset == 'coco' else VOC_NAMES

                    for lab in bbox_and_scores:  # 画框!!!
                        for boxes in bbox_and_scores[lab]:
                            x1, y1, x2, y2, score = boxes
                            if score > 0.05:

                                # 画框
                                cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 3)  # 因为x1,y1,x2,y2的类别是<class 'numpy.float32'>，所以得先转成int型

                                score = round(score, 3)  # 保留score小数点前三位

                                # 写类别名和分数
                                cv.putText(image, names[lab] + ' ' + str(score), (int(x1)+2, int(y1)-10), cv.FONT_HERSHEY_DUPLEX, 1,
                                           (255, 255, 255), 2)
                                reports.append("目标物：%s  检测分数为%s 坐标为%s, %s" % (str(names[lab]), str(score), str(int(x2)-int(x1)), str(int(y2)-int(y1))))

                concat = ""
                for i in reports:
                    concat = concat + i + "\n"  # 将所有的输出语句拼接起来

                fig.patch.set_visible(False)

                img = QtGui.QImage(image, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
                img = QtGui.QPixmap(img).scaled(self.label2.width(), self.label2.height())
                self.label2.setPixmap(img)

                infos = ["所需检测的声呐图像尺寸为:%sx%s" % (str(height), str(width))
                         ]  # 删掉了u

                # reports = ["声呐图像检测结果如下:目标物体类别为,检测分数为%s" % str(score)
                #            ]

                self.wenben5.setText(random.choice(infos))
                self.wenben5.setGeometry(170, 200, 400, 80)
                self.wenben5.setStyleSheet("QLabel{color:white;font-size:15px;font-weight:white;font-family:黑体;}")
                self.wenben6.setText(concat)
                self.wenben6.setGeometry(1115, 180, 400, 200)
                self.wenben6.setStyleSheet("QLabel{color:white;font-size:15px;font-weight:white;font-family:黑体;}")

    def start_image2(self):
        self.timer.start(15, self)

    # 加载模型权重
    def load_model(self):
        if 'hourglass' in cfg.arch:
            model = get_hourglass[cfg.arch]
        # elif 'resdcn' in cfg.arch:
        #   model = get_pose_net(num_layers=int(cfg.arch.split('_')[-1]),
        #                        num_classes=80 if cfg.dataset == 'coco' else 20)
        else:
            raise NotImplementedError

        cfg.device = torch.device('cuda')
        self.net = load_model(model, cfg.ckpt_dir)
        self.net = self.net.to(cfg.device)  # 这行会卡住
        self.net.eval()
        print('Finished loading model!')

    def timerEvent(self, e):

        if self.step >= 100:
            self.step = 0
            self.pbar.setValue(self.step)
            self.timer.stop()
            return
        self.step = self.step + 1
        self.pbar.setValue(self.step)

    def doAction(self, value):
        print("do action")
        if self.timer.isActive():
            self.timer.stop()

        else:
            self.timer.start(100, self)


def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = MainUi()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='centernet')

    parser.add_argument('--root_dir', type=str, default='./')
    parser.add_argument('--img_dir', type=str, default='./data/imgnew0/log_2021-03-15-104853_annotaion_1228.bmp')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/hourglass_addloss_cut_1.5/small_hourglass_epoch300.t7')

    parser.add_argument('--arch', type=str, default='small_hourglass')

    parser.add_argument('--dataset', type=str, default='pascal')
    parser.add_argument('--img_size', type=int, default=512)

    parser.add_argument('--test_flip', action='store_true')
    parser.add_argument('--test_scales', type=str, default='1')  # 0.5,0.75,1,1.25,1.5

    parser.add_argument('--test_topk', type=int, default=100)

    cfg = parser.parse_args()

    os.chdir(cfg.root_dir)

    cfg.test_scales = [float(s) for s in cfg.test_scales.split(',')]

    main()