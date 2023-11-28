import os.path as osp
import torch
import torch.utils.data as data
import cv2
import glob
import numpy as np
from pycocotools.coco import COCO
import random
import math

from utils.augmentations import train_aug, val_aug


# Warning, do not use numpy random in PyTorch multiprocessing, or the random result will be the same.

def train_collate(batch):
    imgs, targets, masks = [], [], []
    valid_batch = [aa for aa in batch if aa[0] is not None]
    vb = len(valid_batch)

    lack_len = len(batch) - len(valid_batch)
    # print(vb, lack_len)
    if lack_len > 0:
        for i in range(lack_len):
            # print('now append:', i)
            valid_batch.append(valid_batch[i])
        # print('valid batch:',len(valid_batch))

    for sample in valid_batch:
        imgs.append(torch.tensor(sample[0], dtype=torch.float32))
        targets.append(torch.tensor(sample[1], dtype=torch.float32))
        masks.append(torch.tensor(sample[2], dtype=torch.float32))

    return torch.stack(imgs, 0), targets, masks


def val_collate(batch):
    imgs = torch.tensor(batch[0][0], dtype=torch.float32).unsqueeze(0)
    targets = torch.tensor(batch[0][1], dtype=torch.float32)
    masks = torch.tensor(batch[0][2], dtype=torch.float32)
    return imgs, targets, masks, batch[0][3], batch[0][4]


def detect_collate(batch):
    imgs = torch.tensor(batch[0][0], dtype=torch.float32).unsqueeze(0)
    return imgs, batch[0][1], batch[0][2]


def detect_onnx_collate(batch):
    return batch[0][0][None, :], batch[0][1], batch[0][2]

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
            (w2 > wh_thr)
            & (h2 > wh_thr)
            & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
            & (ar < ar_thr)
    )  # candidates

def random_perspective(
        img, targets=(), labels=(), masks=(), degrees=10, translate=0.1, scale=[0.1,2], shear=2.0, perspective=0.0, border=(0, 0),
):
    # targets = [cls, xyxy]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * width)  # x translation (pixels)
    T[1, 2] = (random.uniform(0.5 - translate, 0.5 + translate) * height)  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

    ###########################
    # For Aug out of Mosaic
    # s = 1.
    # M = np.eye(3)
    ###########################

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            masks_ = []
            for m in masks:
                m = cv2.warpAffine(m, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
                masks_.append(m)
            masks = np.asarray(masks_)
    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, :4].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, :4] = xy[i]
        targets = targets[:, 1:4]
        masks=masks[i]
        labels=labels[i]
        
    return img, targets, labels, masks

def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    
    return bbox

class COCODetectionHank(data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.mode = mode
        self.cfg = cfg
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.mixup_scale = (0.5, 1.5)
        self.tracking = False
        
        if mode in ('train', 'val'):
            self.image_path = cfg.train_imgs if mode == 'train' else cfg.val_imgs
            self.coco = COCO(cfg.train_ann if mode == 'train' else cfg.val_ann)
            self.ids = list(self.coco.imgToAnns.keys())
        elif mode == 'detect':
            self.image_path = glob.glob(cfg.image + '/*.jpg')
            self.image_path.sort()

        self.continuous_id = cfg.continuous_id
        
    def pull_item(self, index):

        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        # 'target' includes {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
        target = self.coco.loadAnns(ann_ids)
        # target = [aa for aa in target if not aa['iscrowd']]
        target = [aa for aa in target if aa['iscrowd']]

        file_name = self.coco.loadImgs(img_id)[0]['file_name']

        img_path = osp.join(self.image_path, file_name)
        
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        
        box_list, mask_list, label_list = [], [], []

        for aa in target:
            bbox = aa['bbox']

            # When training, some boxes are wrong, ignore them.
            if self.mode == 'train':
                if bbox[0] < 0 or bbox[1] < 0 or bbox[2] < 4 or bbox[3] < 4:
                    continue

            x1y1x2y2_box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
#            x1y1x2y2_box = np.array(bbox)
            category = self.continuous_id[aa['category_id']] - 1

            box_list.append(x1y1x2y2_box)
            mask_list.append(self.coco.annToMask(aa))
            label_list.append(category)    

        if len(box_list) > 0:
            boxes = np.array(box_list)
            masks = np.stack(mask_list, axis=0)
            labels = np.array(label_list)
            assert masks.shape == (boxes.shape[0], height, width), 'Unmatched annotations.'
            
        return img, boxes, labels, masks

    def get_mosaic_coordinate(self, mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
        # TODO update doc
        # index0 to top left part of image
        if mosaic_index == 0:
            x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
            small_coord = w - (x2 - x1), h - (y2 - y1), w, h
        # index1 to top right part of image
        elif mosaic_index == 1:
            x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
            small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
        # index2 to bottom left part of image
        elif mosaic_index == 2:
            x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
            small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
        # index2 to bottom right part of image
        elif mosaic_index == 3:
            x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
            small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
        return (x1, y1, x2, y2), small_coord
    
    def mosaic(self, input_img, idx):
        
        mosaic_boxes = []
        mosaic_labels = []
        input_h, input_w = input_img.shape[:2]

        # yc, xc = s, s  # mosaic center x, y
        yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
        xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

        # 3 additional image indices
        indices = [idx] + [random.randint(0, len(self.ids) - 1) for _ in range(3)]

        for i_mosaic, index in enumerate(indices):
            img, _boxes, _labels, masks = self.pull_item(index)
            h0, w0 = img.shape[:2]  # orig hw
            scale = min(1. * input_h / h0, 1. * input_w / w0)
            img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
#            print('masks_1: ', masks.shape)
            masks = cv2.resize(np.transpose(masks, (1,2,0)), (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
#            print('masks_2: ', masks.shape)
            if len(masks.shape) < 3:
                masks = torch.tensor(masks).unsqueeze(2).cpu().detach().numpy()
            masks = np.transpose(masks, (2,0,1))
#            print('masks_3: ', masks.shape)
            # generate output mosaic image
            (h, w, c) = img.shape[:3]
            if i_mosaic == 0:
                mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)
                mosaic_masks = np.full((masks.shape[0], input_h * 2, input_w * 2), 0, dtype=np.uint8)
            
            else:
                tmp_mosaic_masks = np.full((masks.shape[0], input_h * 2, input_w * 2), 0, dtype=np.uint8)
            
            # suffix l means large image, while s means small image in mosaic aug.
            (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = self.get_mosaic_coordinate(
                mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
            )
            for m in range(masks.shape[0]):   
                
                if i_mosaic == 0:                    
                    mosaic_masks[m][l_y1:l_y2, l_x1:l_x2] = masks[m][s_y1:s_y2, s_x1:s_x2]
                    
                else:
                    
                    try:
                        tmp_mosaic_masks[m][l_y1:l_y2, l_x1:l_x2] = masks[m][s_y1:s_y2, s_x1:s_x2]
                    except:
                        print('tmp_mosaic_masks[m]: ', tmp_mosaic_masks[m].shape)
                        print('[l_y1:l_y2, l_x1:l_x2]: ', (l_y1, l_y2, l_x1, l_x2))
                        print('masks[m]: ', masks[m].shape)
                        print('[s_y1:s_y2, s_x1:s_x2]: ', (s_y1, s_y2, s_x1, s_x2))
                        tmp_mosaic_masks[m][l_y1:l_y2, l_x1:l_x2] = masks[m][s_y1:s_y2, s_x1:s_x2]
#                        continue
            
            mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]

            if i_mosaic != 0:                 
                mosaic_masks = np.concatenate((mosaic_masks, tmp_mosaic_masks))
                    
            padw, padh = l_x1 - s_x1, l_y1 - s_y1

            boxes = _boxes.copy()
            # Normalized xywh to pixel xyxy format
            if _boxes.size > 0:
                boxes[:, 0] = scale * _boxes[:, 0] + padw
                boxes[:, 1] = scale * _boxes[:, 1] + padh
                boxes[:, 2] = scale * _boxes[:, 2] + padw
                boxes[:, 3] = scale * _boxes[:, 3] + padh
            
            mosaic_boxes.append(boxes)
            
            for label in _labels:
                mosaic_labels.append(label)

        if len(mosaic_boxes):
            mosaic_boxes = np.concatenate(mosaic_boxes, 0)
            np.clip(mosaic_boxes[:, 0], 0, 2 * input_w, out=mosaic_boxes[:, 0])
            np.clip(mosaic_boxes[:, 1], 0, 2 * input_h, out=mosaic_boxes[:, 1])
            np.clip(mosaic_boxes[:, 2], 0, 2 * input_w, out=mosaic_boxes[:, 2])
            np.clip(mosaic_boxes[:, 3], 0, 2 * input_h, out=mosaic_boxes[:, 3])
            
        mosaic_labels = np.asarray(mosaic_labels)
        
        return mosaic_img, mosaic_boxes, mosaic_labels, mosaic_masks

    def mixup(self, origin_img, origin_boxes, origin_labels, origin_masks):
        
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_index = random.randint(0, len(self.ids) - 1)
        img, cp_boxes, cp_labels, masks = self.pull_item(cp_index)
        input_dim = origin_img.shape[:2]

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114
            
        cp_masks = np.ones((masks.shape[0], input_dim[0], input_dim[1]), dtype=np.uint8) * 0
            
        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        resized_masks_list = []
        for m in range(masks.shape[0]):
            cp_mask = cv2.resize(
                masks[m],
                (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            )
            resized_masks_list.append(cp_mask)
        resized_masks = np.asarray(resized_masks_list)
        
        cp_img[
        : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img
        cp_masks[:,
        : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_masks
        
        cp_h, cp_w = cp_img.shape[:2]
        cp_img = cv2.resize(
            cp_img,
            (int(cp_w * jit_factor), int(cp_h * jit_factor)),
        )        
        cp_masks_list = []
        for m in range(cp_masks.shape[0]):
            cp_mask = cv2.resize(
                cp_masks[m],
                (int(cp_w * jit_factor), int(cp_h * jit_factor)),
            )
            cp_masks_list.append(cp_mask)
        cp_masks = np.asarray(cp_masks_list)
        
        cp_scale_ratio *= jit_factor
        if FLIP:
            cp_img = cp_img[:, ::-1, :]
            cp_masks = cp_masks[:, :, ::-1]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        
        padded_masks = np.zeros(
            (cp_masks.shape[0], max(origin_h, target_h), max(origin_w, target_w)), dtype=np.uint8
        )
        
        padded_img[:origin_h, :origin_w] = cp_img
        
#        print('cp_masks: ', cp_masks.shape)
#        print('padded_masks: ', padded_masks.shape)
        padded_masks[:, :origin_h, :origin_w] = cp_masks

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
                             y_offset: y_offset + target_h, x_offset: x_offset + target_w
                             ]
        padded_cropped_masks = padded_masks[
                             :, y_offset: y_offset + target_h, x_offset: x_offset + target_w
                             ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_boxes[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                    origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )
        keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)

        if keep_list.sum() >= 1.0:
            cls_labels = cp_labels[keep_list].copy()
            box_labels = cp_bboxes_transformed_np[keep_list].copy()
            mask_labels = padded_cropped_masks[keep_list].copy()
#            if self.tracking:
#                tracking_id_labels = cp_boxes[keep_list, 5:6].copy()
#                labels = np.hstack((box_labels, cls_labels, tracking_id_labels))
#            else:
#                labels = np.hstack((box_labels, cls_labels))
            origin_labels = np.concatenate((origin_labels, cls_labels))
            origin_boxes = np.vstack((origin_boxes, box_labels))
            origin_masks = np.vstack((origin_masks, mask_labels))
            
            origin_img = origin_img.astype(np.float32)
            origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)
            origin_img = origin_img.astype(np.uint8)
            
        return origin_img, origin_boxes, origin_labels, origin_masks
    
    def __getitem__(self, index):
        if self.mode == 'detect':
            img_name = self.image_path[index]
            img_origin = cv2.imread(img_name)
            img_normed = val_aug(img_origin, self.cfg.img_size)
            return img_normed, img_origin, img_name.split(osp.sep)[-1]
        else:
            img_id = self.ids[index]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)

            # 'target' includes {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
            target = self.coco.loadAnns(ann_ids)
            # target = [aa for aa in target if not aa['iscrowd']]
            target = [aa for aa in target if aa['iscrowd']]

            file_name = self.coco.loadImgs(img_id)[0]['file_name']

            img_path = osp.join(self.image_path, file_name)
            assert osp.exists(img_path), f'Image path does not exist: {img_path}'

            img = cv2.imread(img_path)
            height, width, _ = img.shape

            assert len(target) > 0, 'No annotation in this image!'
            box_list, mask_list, label_list = [], [], []

            for aa in target:
                bbox = aa['bbox']

                # When training, some boxes are wrong, ignore them.
                if self.mode == 'train':
                    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] < 4 or bbox[3] < 4:
                        continue

                x1y1x2y2_box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                category = self.continuous_id[aa['category_id']] - 1

                box_list.append(x1y1x2y2_box)
                mask_list.append(self.coco.annToMask(aa))
                label_list.append(category)

            if len(box_list) > 0:
                boxes = np.array(box_list)
                masks = np.stack(mask_list, axis=0)
                labels = np.array(label_list)
                assert masks.shape == (boxes.shape[0], height, width), 'Unmatched annotations.'

                if self.mode == 'train':
                    
                    # '''mosaic'''
                    # if random.random() < self.mosaic_prob:
                    #     img, boxes, labels, masks = self.mosaic(img, index)
                    
                    ''''random_perspective'''
#                    img, boxes, labels, masks = random_perspective(
#                                                                img,
#                                                                np.concatenate((labels[:, np.newaxis], boxes),1),
#                                                                labels,
#                                                                masks,
#                                                                border=[-height // 2, -width // 2],
#                                                            )  #
#  
                    
                    # '''mixup'''
                    # if random.random() < self.mixup_prob:
                    #     img, boxes, labels, masks = self.mixup(img, boxes, labels, masks)
                        
                    img, masks, boxes, labels = train_aug(img, masks, boxes, labels, self.cfg.img_size)
                    if img is None:
                        return None, None, None
                    else:
                        boxes = np.hstack((boxes, np.expand_dims(labels, axis=1)))
                        return img, boxes, masks
                elif self.mode == 'val':
                    img = val_aug(img, self.cfg.img_size)
                    boxes = boxes / np.array([width, height, width, height])  # to 0~1 scale
                    boxes = np.hstack((boxes, np.expand_dims(labels, axis=1)))
                    return img, boxes, masks, height, width
            else:
                if self.mode == 'val':
                    raise RuntimeError('Error, no valid object in this image.')
                else:
                    print(f'No valid object in image: {img_id}. Use a repeated image in this batch.')
                    return None, None, None

    def __len__(self):
        if self.mode == 'train':
            return len(self.ids)
        elif self.mode == 'val':
            return len(self.ids) if self.cfg.val_num == -1 else min(self.cfg.val_num, len(self.ids))
        elif self.mode == 'detect':
            return len(self.image_path)
        
class COCODetection(data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.mode = mode
        self.cfg = cfg

        if mode in ('train', 'val'):
            self.image_path = cfg.train_imgs if mode == 'train' else cfg.val_imgs
            self.coco = COCO(cfg.train_ann if mode == 'train' else cfg.val_ann)
            self.ids = list(self.coco.imgToAnns.keys())
        elif mode == 'detect':
            self.image_path = glob.glob(cfg.image + '/*.jpg')
            self.image_path.sort()

        self.continuous_id = cfg.continuous_id

    def __getitem__(self, index):
        if self.mode == 'detect':
            img_name = self.image_path[index]
            img_origin = cv2.imread(img_name)
            img_normed = val_aug(img_origin, self.cfg.img_size)
            return img_normed, img_origin, img_name.split(osp.sep)[-1]
        else:
            img_id = self.ids[index]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)

            # 'target' includes {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
            target = self.coco.loadAnns(ann_ids)
            #target = [aa for aa in target if not aa['iscrowd']]
            target = [aa for aa in target if aa['iscrowd']]
            file_name = self.coco.loadImgs(img_id)[0]['file_name']

            img_path = osp.join(self.image_path, file_name)
            assert osp.exists(img_path), f'Image path does not exist: {img_path}'

            img = cv2.imread(img_path)
            height, width, _ = img.shape
            assert len(target) > 0, 'No annotation in this image!'
            box_list, mask_list, label_list = [], [], []

            for aa in target:
                bbox = aa['bbox']

                # When training, some boxes are wrong, ignore them.
                if self.mode == 'train':
                    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] < 4 or bbox[3] < 4:
                        continue

                x1y1x2y2_box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                category = self.continuous_id[aa['category_id']] - 1

                box_list.append(x1y1x2y2_box)
                mask_list.append(self.coco.annToMask(aa))
                label_list.append(category)

            if len(box_list) > 0:
                boxes = np.array(box_list)
                masks = np.stack(mask_list, axis=0)
                labels = np.array(label_list)
                assert masks.shape == (boxes.shape[0], height, width), 'Unmatched annotations.'

                if self.mode == 'train':
                    img, masks, boxes, labels = train_aug(img, masks, boxes, labels, self.cfg.img_size)
                    if img is None:
                        return None, None, None
                    else:
                        boxes = np.hstack((boxes, np.expand_dims(labels, axis=1)))
                        return img, boxes, masks
                elif self.mode == 'val':
                    img = val_aug(img, self.cfg.img_size)
                    boxes = boxes / np.array([width, height, width, height])  # to 0~1 scale
                    boxes = np.hstack((boxes, np.expand_dims(labels, axis=1)))
                    return img, boxes, masks, height, width
            else:
                if self.mode == 'val':
                    raise RuntimeError('Error, no valid object in this image.')
                else:
                    print(f'No valid object in image: {img_id}. Use a repeated image in this batch.')
                    return None, None, None

    def __len__(self):
        if self.mode == 'train':
            return len(self.ids)
        elif self.mode == 'val':
            return len(self.ids) if self.cfg.val_num == -1 else min(self.cfg.val_num, len(self.ids))
        elif self.mode == 'detect':
            return len(self.image_path)
