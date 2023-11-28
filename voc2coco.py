# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 03:40:28 2022

@author: 3090
"""

#%%
import os
import numpy as np
import json
from detectron2.structures import BoxMode
import scipy.io
import cv2
import pycocotools.mask
#from imantics import Polygons, Mask
from shapely.geometry import Polygon, MultiPolygon
from skimage import measure         

PASCAL_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def mask2bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return cmin, rmin, cmax - cmin, rmax - rmin

def get_voc2coco_dicts(folder_path, file_type):
    
    
    img_path = folder_path + 'img/'
    inst_path = folder_path + 'inst/'
    
    img_name_fmt = '%s.jpg'
    ann_name_fmt = '%s.mat'
    
    with open(folder_path + f'/{file_type}.txt', 'r') as f:
        names = f.read().strip().split('\n')

    dataset_dicts = []
    for i, name in enumerate(names):
        
        record = {}
        
        filename = img_path + img_name_fmt % name
        ann_path = inst_path + ann_name_fmt % name
        
        ann = scipy.io.loadmat(ann_path)['GTinst'][0][0]
        classes = [int(x[0]) for x in ann[2]]
        seg = ann[0]
        
        record['file_name'] = filename
        
        img = cv2.imread(filename)
        
        record['height'] = img.shape[0]
        record['width'] = img.shape[1]
        
        classes = [int(x[0]) for x in ann[2]-1]
                
        objs = []
        for idx in range(len(classes)):
            
            mask = (seg == (idx + 1)).astype(np.float)
             
#            masks = create_sub_masks(mask)
            
            contours = measure.find_contours(mask, 0.5, positive_orientation='low')
            
#            rle = pycocotools.mask.encode(np.asfortranarray(mask.astype(np.uint8)))
#            rle['counts'] = rle['counts'].decode('ascii')
                
#            polygons = Mask(mask).polygons()
            
            segmentations = []
            polygons = []
            
            try:
                for contour in contours:
                    # Flip from (row, col) representation to (x, y)
                    # and subtract the padding pixel
                    for i in range(len(contour)):
                        row, col = contour[i]
                        contour[i] = (col - 1, row - 1)
            
                    # Make a polygon and simplify it
                    poly = Polygon(contour)
                        
                    poly = poly.simplify(1.0, preserve_topology=True)
                    polygons.append(poly)
                    segmentation = np.array(poly.exterior.coords).ravel().tolist()
                    segmentations.append(segmentation)
            
            except:
                print('file_name: ', filename)
                print(contour)
                continue
                    
            multi_poly = MultiPolygon(polygons)
            x, y, max_x, max_y = multi_poly.bounds
            width = max_x - x
            height = max_y - y
            bbox = (x, y, width, height)
            area = multi_poly.area

            obj = {
#                    'bbox':[int(x) for x in mask2bbox(mask)],
                    'bbox': bbox,
                    'area': area,
                    'bbox_mode': BoxMode.XYWH_ABS,
#                    'segmentation':rle,
#                    'segmentation':polygons.segmentation,
                    'segmentation':segmentations,
                    'category_id':classes[idx],
                    'iscrowd':0
                    }
            objs.append(obj)
        
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts

from detectron2.data import DatasetCatalog, MetadataCatalog
for d in ["train", "val"]:
    DatasetCatalog.register("voc2coco_" + d, lambda d=d: get_voc2coco_dicts('C:/Users/User/Hank/yolact-master/data/sbd/', d))
    MetadataCatalog.get("voc2coco_" + d).set(thing_classes=PASCAL_CLASSES)
microcontroller_metadata = MetadataCatalog.get("voc2coco_train")
#%%
from PIL import Image

def create_sub_masks(mask_image):
    mask_image = Image.fromarrary(mask_image*255)
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))#[1:]

            # If the pixel is not black...
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)
                
    return sub_masks
#%%
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("voc2coco_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
cfg.INPUT.MASK_FORMAT = 'bitmask'

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
