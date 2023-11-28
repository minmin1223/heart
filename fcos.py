#%%
#import sys
#sys.path.append('/Users/hank/Downloads/fcos-pytorch-master/fcos-pytorch-master')
import math

import torch
from torch import nn
from torch.nn import functional as F

from loss import FCOSLoss
from postprocess import FCOSPostprocessor


class Scale(nn.Module):
    def __init__(self, init=1.0):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor([init], dtype=torch.float32))

    def forward(self, input):
        return input * self.scale


def init_conv_kaiming(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight, a=1)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_conv_std(module, std=0.01):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, std=std)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class FPN(nn.Module):
    def __init__(self, in_channels, out_channel, top_blocks=None):
        super().__init__()

        self.inner_convs = nn.ModuleList()
        self.out_convs = nn.ModuleList()

        for i, in_channel in enumerate(in_channels, 1):
            if in_channel == 0:
                self.inner_convs.append(None)
                self.out_convs.append(None)

                continue

            inner_conv = nn.Conv2d(in_channel, out_channel, 1)
            feat_conv = nn.Conv2d(out_channel, out_channel, 3, padding=1)

            self.inner_convs.append(inner_conv)
            self.out_convs.append(feat_conv)

        self.apply(init_conv_kaiming)

        self.top_blocks = top_blocks

    def forward(self, inputs):
        inner = self.inner_convs[-1](inputs[-1])
        outs = [self.out_convs[-1](inner)]
        
#        for i in inputs:
#            print(i.shape)
            
        for feat, inner_conv, out_conv in zip(
            inputs[:-1][::-1], self.inner_convs[:-1][::-1], self.out_convs[:-1][::-1]
        ):
#            print('inner_conv: ', inner_conv)
#            print('feat: ', feat.shape)
            if inner_conv is None:
                continue
            
#            print('pre up inner: ', inner.shape)
            upsample = F.interpolate(inner, scale_factor=2, mode='nearest')
            inner_feat = inner_conv(feat)
            inner = inner_feat + upsample
            outs.insert(0, out_conv(inner))
            
#            print('post up input: ', inner.shape)
#            print('upsample: ', upsample.shape)
#            print('inner_feat: ', inner_feat.shape)
#            
#            print('===========================================================')
#            for i in outs:
#                print('outs: ', i.shape)
#            print('===========================================================')

        if self.top_blocks is not None:
            top_outs = self.top_blocks(outs[-1], inputs[-1])
            outs.extend(top_outs)
        
#        for i in outs:
#            print(i.shape)
            
        return outs


class FPNTopP6P7(nn.Module):
    def __init__(self, in_channel, out_channel, use_p5=True):
        super().__init__()

        self.p6 = nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1)
        self.p7 = nn.Conv2d(out_channel, out_channel, 3, stride=2, padding=1)

        self.apply(init_conv_kaiming)

        self.use_p5 = use_p5

    def forward(self, f5, p5):
        input = p5 if self.use_p5 else f5

        p6 = self.p6(input)
        p7 = self.p7(F.relu(p6))

        return p6, p7


class FCOSHead(nn.Module):
    def __init__(self, in_channel, n_class, n_conv, prior):
        super().__init__()

        n_class = n_class - 1

        cls_tower = []
        bbox_tower = []

        for i in range(n_conv):
            cls_tower.append(
                nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)
            )
            cls_tower.append(nn.GroupNorm(32, in_channel))
            cls_tower.append(nn.ReLU())

            bbox_tower.append(
                nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)
            )
            bbox_tower.append(nn.GroupNorm(32, in_channel))
            bbox_tower.append(nn.ReLU())

        self.cls_tower = nn.Sequential(*cls_tower)
        self.bbox_tower = nn.Sequential(*bbox_tower)

        self.cls_pred = nn.Conv2d(in_channel, n_class, 3, padding=1)
        self.ins_pred = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, 3, padding=1),
                nn.Tanh()
                )
        self.bbox_pred = nn.Conv2d(in_channel, 4, 3, padding=1)
        self.center_pred = nn.Conv2d(in_channel, 1, 3, padding=1)

        self.apply(init_conv_std)

        prior_bias = -math.log((1 - prior) / prior)
        nn.init.constant_(self.cls_pred.bias, prior_bias)

        self.scales = nn.ModuleList([Scale(1.0) for _ in range(5)])

    def forward(self, input):
        logits = []
        bboxes = []
        centers = []
        insts = []

        for feat, scale in zip(input, self.scales):
            cls_out = self.cls_tower(feat)

            logits.append(self.cls_pred(cls_out))
            centers.append(self.center_pred(cls_out))
            insts.append(self.ins_pred(cls_out))

            bbox_out = self.bbox_tower(feat)
            bbox_out = torch.exp(scale(self.bbox_pred(bbox_out)))

            bboxes.append(bbox_out)

        return logits, bboxes, centers, insts

class FCOSdeHead(nn.Module):
    def __init__(self, in_channel, n_class, n_conv, prior, levels=5):
        super().__init__()

        n_class = n_class - 1
        
        self.cls_preds = nn.ModuleList()
        self.center_preds = nn.ModuleList()
        self.bbox_preds = nn.ModuleList()
        self.stem_preds = nn.ModuleList()
        self.ins_preds = nn.ModuleList()
        
        for i in range(levels):
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256),
                    out_channels=n_class,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            self.center_preds.append(
                nn.Conv2d(
                    in_channels=int(256),
                    out_channels=1,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            self.bbox_preds.append(
                nn.Conv2d(
                    in_channels=int(256),
                    out_channels=4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            ) 
            self.stem_preds.append(
                nn.Sequential(
                nn.Conv2d(
                    in_channels=int(256),
                    out_channels=int(256),
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
                
            )
            ) 
            self.ins_preds.append(
                nn.Conv2d(
                    in_channels=int(256),
                    out_channels=n_class,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            ) 
                
#        self.apply(init_conv_std)

#        prior_bias = -math.log((1 - prior) / prior)
#        nn.init.constant_(self.cls_pred.bias, prior_bias)

        self.scales = nn.ModuleList([Scale(1.0) for _ in range(5)])

    def forward(self, input):
        logits = []
        bboxes = []
        centers = []
        insts = []

        for i, (feat, scale) in enumerate(zip(input, self.scales)):
#            cls_out = self.cls_tower(feat)
            feat = self.stem_preds[i](feat)
            logits.append(self.cls_preds[i](feat))
            centers.append(self.center_preds[i](feat))
            insts.append(self.ins_preds[i](feat))

            bbox_out = torch.exp(scale(self.bbox_preds[i](feat)))
            bboxes.append(bbox_out)

        return logits, bboxes, centers, insts
#class Ins_head(nn.module):
#    def __init__(self, channels):=
#    
#        ins_convs = nn.Conv2d

class FCOS(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
#        fpn_top = FPNTopP6P7(
#            config.feat_channels[-1], config.out_channel, use_p5=config.use_p5
#        )
        fpn_top = FPNTopP6P7(
            1024, 256, True
        )
#        self.fpn = FPN(config.feat_channels, config.out_channel, fpn_top)
        self.fpn = FPN(
                [0, 0, 256, 512, 1024],
                256,
                fpn_top)
#        self.head = FCOSHead(
#            config.out_channel, config.n_class, config.n_conv, config.prior
#        )
        self.head = FCOSHead(
            256,
            21,
            4,
            0.01
        )
        
#        self.postprocessor = FCOSPostprocessor(
#            config.threshold,       # 0.05
#            config.top_n,           # 1000
#            config.nms_threshold,   # 0.6
#            config.post_top_n,      # 100
#            config.min_size,        # 0
#            config.n_class,         # 81
#        )
        self.postprocessor = FCOSPostprocessor(
                threshold = 0.05,       # 0.05
                top_n = 1000,           # 1000
                nms_threshold = 0.6,   # 0.6
                post_top_n = 100,      # 100
                min_size = 0,        # 0
                n_class = 21,         # 81
            )
        
#        self.loss = FCOSLoss(
#            config.sizes,           # [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 100000000]]
#            config.gamma,           # 2.0
#            config.alpha,           # 0.25
#            config.iou_loss_type,   # giou
#            config.center_sample,   # True
#            config.fpn_strides,     # [8, 16, 32, 64, 128]
#            config.pos_radius,      # 1.5
#        )
        self.loss = FCOSLoss(
                sizes = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 100000000]],           # [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 100000000]]
                gamma = 2.0,           # 2.0
                alpha = 0.25,           # 0.25
                iou_loss_type = 'giou',   # giou
                center_sample = True,   # True
                fpn_strides = [8, 16, 32, 64, 128],     # [8, 16, 32, 64, 128]
                pos_radius = 1.5,      # 1.5
            )

#        self.fpn_strides = config.fpn_strides
        self.fpn_strides = [8, 16, 32, 64, 128]
        self.semantic_seg_conv = nn.Conv2d(256, 20, kernel_size=1)
        self.can_conv = nn.Conv2d(256, 256, kernel_size=1)
        
    def train(self, mode=True):
        super().train(mode)

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

        self.apply(freeze_bn)

    def forward(self, input, image_sizes=None, targets=None, masks=None):

#        x = torch.Size([1, 3, 544, 544])
        
        # VoVNet
        features = self.backbone(input)
#        torch.Size([1, 64, 272, 272])
#        torch.Size([1, 256, 136, 136])
#        torch.Size([1, 512, 68, 68])
#        torch.Size([1, 768, 34, 34])
#        torch.Size([1, 1024, 17, 17])
#        for i in features:
#            print(i.shape)
        
        # FPN
        features = self.fpn(features)
#        torch.Size([1, 256, 68, 68])
#        torch.Size([1, 256, 34, 34])
#        torch.Size([1, 256, 17, 17])
#        torch.Size([1, 256, 9, 9])
#        torch.Size([1, 256, 5, 5])
        
        # FCOS Head
        cls_pred, box_pred, center_pred, ins_preds = self.head(features[:])        
#        cls_pred:                      box_pred:                   center_pred:
#        torch.Size([1, 80, 68, 68])    torch.Size([1, 4, 68, 68])  torch.Size([1, 1, 68, 68])
#        torch.Size([1, 80, 34, 34])    torch.Size([1, 4, 34, 34])  torch.Size([1, 1, 34, 34])
#        torch.Size([1, 80, 17, 17])    torch.Size([1, 4, 17, 17])  torch.Size([1, 1, 17, 17])
#        torch.Size([1, 80, 9, 9])      torch.Size([1, 4, 9, 9])    torch.Size([1, 1, 9, 9])
#        torch.Size([1, 80, 5, 5])      torch.Size([1, 4, 5, 5])    torch.Size([1, 1, 5, 5])
    
        # print(cls_pred, box_pred, center_pred)
        location = self.compute_location(features[:])  
        
#        FPN features ->
#        torch.Size([4624, 2])
#        torch.Size([1156, 2])
#        torch.Size([289, 2])
#        torch.Size([81, 2])
#        torch.Size([25, 2])
        
        seg_pred = self.semantic_seg_conv(features[0]) 
        can_pred = self.can_conv(features[0])
#        seg_pred = features[0]
#        return location, cls_pred, box_pred, center_pred, targets
        
        ins_pred = []
        for i in range(len(ins_preds)):
#            print(ins_preds[i].shape)
            ins_pred.append(ins_preds[i].permute(0,2,3,1).reshape(ins_preds[0].shape[0], -1, 256))
        
        ins_pred = torch.cat(ins_pred, 1)
            
        if self.training:
        
            
#            return features, location, cls_pred, box_pred, center_pred, seg_pred, ins_pred, targets, masks
        
            loss_cls, loss_box, loss_center, loss_ins, loss_seg = self.loss(
                location, cls_pred, box_pred, center_pred, seg_pred, ins_pred, can_pred, targets, masks
            )
            
#            class_gt = [None] * len(targets)
#            batch_size = input.size(0)
#            for i in range(batch_size):
#                class_gt[i] = targets[i][:, -1].long()   
#            loss_seg = self.semantic_seg_loss(seg_pred,  masks, class_gt)
            
            losses = {
                'loss_cls': loss_cls,
                'loss_box': loss_box,
                'loss_center': loss_center,
                'loss_seg': loss_seg,
                'loss_ins': loss_ins
            }

            return features, losses

        else:
#            loss_cls, loss_box, loss_center, loss_ins, loss_seg = self.loss(
#                location, cls_pred, box_pred, center_pred, seg_pred, ins_pred, targets, masks
#            )
#            
#            losses = {
#                'loss_cls': loss_cls,
#                'loss_box': loss_box,
#                'loss_center': loss_center,
#                'loss_seg': loss_seg,
#                'loss_ins': loss_ins
#            }
#            print(seg_pred.shape)
#            return location, cls_pred, box_pred, center_pred, seg_pred, ins_pred#, targets, masks
            score_pred, class_pred, box_pred, seg_pred = self.postprocessor(
                location, cls_pred, box_pred, center_pred, seg_pred, ins_pred, can_pred, image_sizes
            )
#            print('seg_pred: ', seg_pred.shape)
            
            return score_pred, class_pred, box_pred, seg_pred#, losses

    def compute_location(self, features):
        locations = []

        for i, feat in enumerate(features):
            _, _, height, width = feat.shape
            location_per_level = self.compute_location_per_level(
                height, width, self.fpn_strides[i], feat.device
            )
            locations.append(location_per_level)

        return locations

    def compute_location_per_level(self, height, width, stride, device):
        shift_x = torch.arange(
            0, width * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y = torch.arange(
            0, height * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        location = torch.stack((shift_x, shift_y), 1) + stride // 2

        return location

#    def semantic_seg_loss(self, segmentation_p, mask_gt, class_gt):
#        # Note classes here exclude the background class, so num_classes = cfg.num_classes - 1
#        batch_size, num_classes, mask_h, mask_w = segmentation_p.size()
#        loss_s = 0
#
#        for i in range(batch_size):
#            cur_segment = segmentation_p[i]
#            cur_class_gt = class_gt[i]
#
#            downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0), (mask_h, mask_w), mode='bilinear',
#                                              align_corners=False).squeeze(0)
#            downsampled_masks = downsampled_masks.gt(0.5).float()
#
#            # Construct Semantic Segmentation
#            segment_gt = torch.zeros_like(cur_segment, requires_grad=False)
#            for j in range(downsampled_masks.size(0)):
##                print('segment_gt: ', segment_gt[cur_class_gt[j]].shape)
##                print('downsample_masks: ', downsampled_masks[j].shape)
#                segment_gt[cur_class_gt[j]] = torch.max(segment_gt[cur_class_gt[j]], downsampled_masks[j])
#
#            loss_s += F.binary_cross_entropy_with_logits(cur_segment, segment_gt, reduction='sum')
#        
#        return loss_s / mask_h / mask_w / batch_size
        
