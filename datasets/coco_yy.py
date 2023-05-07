import os
import cv2
import json
import math
import numpy as np

import torch
import torch.utils.data as data
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval

from utils.image import get_border, get_affine_transform, affine_transform, color_aug
from utils.image import gaussian_radius, draw_umich_gaussian

COCO_NAMES = ['__background__', 'ball', 'cylinder', 'square cage', 'cube', 'circle cage', 'human body',
              'metal bucket', 'tyre']

COCO_NAMES_Train = ['__background__', 'ball', 'cylinder', 'square cage', 'cube', 'circle cage', 'human body',
                    'metal bucket', 'tyre', 'ball_shw', 'cylinder_shw', 'square cage_shw', 'cube_shw',
                    'circle cage_shw', 'human body_shw', 'metal bucket_shw', 'tyre_shw']

COCO_MEAN = [0.40789654, 0.44719302, 0.47026115]
COCO_STD = [0.28863828, 0.27408164, 0.27809835]
COCO_EIGEN_VALUES = [0.2141788, 0.01817699, 0.00341571]
COCO_EIGEN_VECTORS = [[-0.58752847, -0.69563484, 0.41340352],
                      [-0.5832747, 0.00994535, -0.81221408],
                      [-0.56089297, 0.71832671, 0.41158938]]


class COCO(data.Dataset):
    def __init__(self, data_dir, split, split_ratio=1, img_size=512):  # split ['train', 'val']
        super(COCO, self).__init__()
        self.num_classes = 8  # numbers of hmap
        self.class_name = COCO_NAMES_Train
        self.valid_ids = np.arange(1, 17, dtype=np.int32)
        self.cat_ids = {v: i for i, v in enumerate(self.valid_ids)}  # 用于取json中的物体label以及转化输出结果标签

        self.data_rng = np.random.RandomState(123)
        self.eig_val = np.array(COCO_EIGEN_VALUES, dtype=np.float32)
        self.eig_vec = np.array(COCO_EIGEN_VECTORS, dtype=np.float32)
        self.mean = np.array(COCO_MEAN, dtype=np.float32).reshape(1, 1, 3)  # 均转化为3维的形式
        self.std = np.array(COCO_STD, dtype=np.float32).reshape(1, 1, 3)  # 均转化为3维的形式
        # self.mean = np.array(COCO_MEAN, dtype=np.float32)[None, None, :]
        # self.std = np.array(COCO_STD, dtype=np.float32)[None, None, :]

        self.split = split
        self.data_dir = os.path.join(data_dir, 'coco')
        self.img_dir = os.path.join(self.data_dir, '%s2017' % split)
        if split in ('test', 'val', 'eval'):
            self.annot_path = os.path.join(self.data_dir, 'annotations', 'image_info_test-dev2017_shw.json')
        else:
            self.annot_path = os.path.join(self.data_dir, 'annotations', 'pascal_train2020_shw.json')

        self.max_objs = 128
        self.padding = 31  # 127 for hourglass # 31 for resnet/resdcn
        self.down_ratio = 4
        self.img_size = {'h': img_size, 'w': img_size}
        self.fmap_size = {'h': img_size // self.down_ratio, 'w': img_size // self.down_ratio}
        self.rand_scales = np.arange(0.6, 1.4, 0.1)
        self.gaussian_iou = 0.7

        print('==> initializing coco 2017 %s data.' % split)
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)
        print('Loaded %d %s samples' % (self.num_samples, split))

        # if 0 < split_ratio < 1:
        #   split_size = int(np.clip(split_ratio * len(self.images), 1, len(self.images)))
        #   self.images = self.images[:split_size]

    def __getitem__(self, index):
        img_id = self.images[index]
        img_path = os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name'])
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        annotations = self.coco.loadAnns(ids=ann_ids)

        labels = []
        bboxes = []
        yy_labels = []
        yy_bboxes = []

        for anno in annotations:
            if int(anno['category_id']) >= 9:
                yy_labels.append(self.cat_ids[anno['category_id']])
                yy_bboxes.append(anno['bbox'])

            else:
                labels.append(self.cat_ids[anno['category_id']])
                bboxes.append(anno['bbox'])

        labels = np.array(labels)
        bboxes = np.array(bboxes, dtype=np.float32)
        yy_labels = np.array(yy_labels)
        yy_bboxes = np.array(yy_bboxes)

        if len(bboxes) == 0:
            bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
            labels = np.array([[0]])

        if len(yy_bboxes) == 0:
            yy_bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
            yy_labels = np.array([[0]])

        bboxes[:, 2:] += bboxes[:, :2]  # xywh to xyxy
        yy_bboxes[:, 2:] += yy_bboxes[:, :2]  # xywh to xyxy

        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        center = np.array([width / 2., height / 2.], dtype=np.float32)  # center of image
        scale = max(height, width) * 1.0

        flipped = False
        if self.split == 'train':
            scale = scale * np.random.choice(self.rand_scales)
            w_border = get_border(128, width)
            h_border = get_border(128, height)
            center[0] = np.random.randint(low=w_border, high=width - w_border)
            center[1] = np.random.randint(low=h_border, high=height - h_border)

            if np.random.random() < 0.5:
                flipped = True
                img = img[:, ::-1, :]
                center[0] = width - center[0] - 1

        trans_img = get_affine_transform(center, scale, 0, [self.img_size['w'], self.img_size['h']])
        img = cv2.warpAffine(img, trans_img, (self.img_size['w'], self.img_size['h']))

        # -----------------------------------debug---------------------------------
        # for bbox, label in zip(bboxes, labels):
        #   if flipped:
        #     bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        #   bbox[:2] = affine_transform(bbox[:2], trans_img)
        #   bbox[2:] = affine_transform(bbox[2:], trans_img)
        #   bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.img_size['w'] - 1)
        #   bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.img_size['h'] - 1)
        #   cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        #   cv2.putText(img, self.class_name[label + 1], (int(bbox[0]), int(bbox[1])),
        #               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # cv2.imshow('img', img)
        # cv2.waitKey()
        # -----------------------------------debug---------------------------------

        img = img.astype(np.float32) / 255.

        if self.split == 'train':
            color_aug(self.data_rng, img, self.eig_val, self.eig_vec)

        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)  # from [H, W, C] to [C, H, W]

        trans_fmap = get_affine_transform(center, scale, 0, [self.fmap_size['w'], self.fmap_size['h']])

        hmap = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)  # heatmap
        w_h_ = np.zeros((self.max_objs, 2), dtype=np.float32)  # width and height
        regs = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression
        inds = np.zeros((self.max_objs,), dtype=np.int64)
        ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)

        # detections = []
        for k, (bbox, label) in enumerate(zip(bboxes, labels)):
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_fmap)
            bbox[2:] = affine_transform(bbox[2:], trans_fmap)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.fmap_size['w'] - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.fmap_size['h'] - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            if h > 0 and w > 0:
                obj_c = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                obj_c_int = obj_c.astype(np.int32)

                radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)), self.gaussian_iou)))
                draw_umich_gaussian(hmap[label], obj_c_int, radius)
                w_h_[k] = 1. * w, 1. * h
                regs[k] = obj_c - obj_c_int  # discretization error
                inds[k] = obj_c_int[1] * self.fmap_size['w'] + obj_c_int[0]
                ind_masks[k] = 1
                # groundtruth bounding box coordinate with class
                # detections.append([obj_c[0] - w / 2, obj_c[1] - h / 2,
                #                    obj_c[0] + w / 2, obj_c[1] + h / 2, 1, label])

        # detections = np.array(detections, dtype=np.float32) \
        #   if len(detections) > 0 else np.zeros((1, 6), dtype=np.float32)

        # 阴影标签
        yy_hmap = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)  # heatmap
        yy_w_h_ = np.zeros((self.max_objs, 2), dtype=np.float32)  # width and height
        yy_regs = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression
        yy_inds = np.zeros((self.max_objs,), dtype=np.int64)
        yy_ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)
        for k, (bbox, label) in enumerate(zip(yy_bboxes, yy_labels)):
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_fmap)
            bbox[2:] = affine_transform(bbox[2:], trans_fmap)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.fmap_size['w'] - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.fmap_size['h'] - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                obj_c = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                obj_c_int = obj_c.astype(np.int32)

                radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)), self.gaussian_iou)))
                draw_umich_gaussian(yy_hmap[label - 8], obj_c_int, radius)
                yy_w_h_[k] = 1. * w, 1. * h
                yy_regs[k] = obj_c - obj_c_int  # discretization error
                yy_inds[k] = obj_c_int[1] * self.fmap_size['w'] + obj_c_int[0]
                yy_ind_masks[k] = 1

        return {'image': img,
                'hmap': hmap, 'w_h_': w_h_, 'regs': regs, 'inds': inds, 'ind_masks': ind_masks,
                'c': center, 's': scale, 'img_id': img_id,
                'yy_hmap': yy_hmap, 'yy_w_h_': yy_w_h_, 'yy_regs': yy_regs, 'yy_inds': yy_inds,
                'yy_ind_masks': yy_ind_masks}

    def __len__(self):
        return self.num_samples


class COCO_eval(COCO):
    def __init__(self, data_dir, split, test_scales=(1,), test_flip=False, fix_size=False, *args, **kwargs):
        super(COCO_eval, self).__init__(data_dir, split, *args, **kwargs)
        self.class_name = COCO_NAMES
        self.test_flip = test_flip
        self.test_scales = test_scales
        self.fix_size = fix_size

    def __getitem__(self, index):
        img_id = self.images[index]
        img_path = os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name'])
        image = cv2.imread(img_path)
        # print(img_path)
        height, width = image.shape[0:2]

        out = {}
        for scale in self.test_scales:
            new_height = int(height * scale)
            new_width = int(width * scale)

            if self.fix_size:
                img_height, img_width = self.img_size['h'], self.img_size['w']
                center = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
                scaled_size = max(height, width) * 1.0
                scaled_size = np.array([scaled_size, scaled_size], dtype=np.float32)
            else:
                img_height = (new_height | self.padding) + 1
                img_width = (new_width | self.padding) + 1
                center = np.array([new_width // 2, new_height // 2], dtype=np.float32)
                scaled_size = np.array([img_width, img_height], dtype=np.float32)

            img = cv2.resize(image, (new_width, new_height))
            trans_img = get_affine_transform(center, scaled_size, 0, [img_width, img_height])
            img = cv2.warpAffine(img, trans_img, (img_width, img_height))

            img = img.astype(np.float32) / 255.
            img -= self.mean
            img /= self.std
            img = img.transpose(2, 0, 1)[None, :, :, :]  # from [H, W, C] to [1, C, H, W]

            if self.test_flip:
                img = np.concatenate((img, img[:, :, :, ::-1].copy()), axis=0)

            out[scale] = {'image': img,
                          'center': center,
                          'scale': scaled_size,
                          'fmap_h': img_height // self.down_ratio,
                          'fmap_w': img_width // self.down_ratio}

        return img_id, out

    def convert_eval_format(self, all_bboxes):
        # all_bboxes: num_samples x num_classes x 5
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self.valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(lambda x: float("{:.2f}".format(x)), bbox[0:4]))

                    detection = {"image_id": int(image_id),
                                 "category_id": int(category_id),
                                 "bbox": bbox_out,
                                 "score": float("{:.2f}".format(score))}
                    detections.append(detection)
        return detections

    def run_eval(self, results, save_dir=None):
        """
    param:
      eval_dataset: is a coco class
    """
        detections = self.convert_eval_format(results)
        if save_dir is not None:
            result_json = os.path.join(save_dir, "results.json")
            json.dump(detections, open(result_json, "w"))

        coco_dets = self.coco.loadRes(detections)  # return a coco class
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        # print("eval_Image:", coco_eval.evalImgs)
        # print("IOU:", coco_eval.ious)
        # print("param: ", coco_eval.params.imgIds,
        #       "\n", coco_eval.params.areaRng, coco_eval.params.areaRngLbl)
        # print("gt:", coco_eval._gts)
        # print("det:", coco_eval._dts)
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats

    @staticmethod
    def collate_fn(batch):
        out = []
        for img_id, sample in batch:
            out.append((img_id, {s: {k: torch.from_numpy(sample[s][k]).float()
            if k == 'image' else np.array(sample[s][k]) for k in sample[s]} for s in sample}))
        return out


if __name__ == '__main__':
    from tqdm import tqdm
    import pickle

    dataset = COCO('/home/icisee/XTW/声学/pytorch_simple_CenterNet_45-master/data/', 'train')
    for d in dataset:
        b1 = d
    #   pass

    pass
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=2,
    #                                            shuffle=False, num_workers=0,
    #                                            pin_memory=True, drop_last=True)
    #
    # for b in tqdm(train_loader):
    #   pass
