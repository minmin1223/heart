import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import sys
#sys.path.append('C:/Users/User/Hank/Yolact_minimal-master')

from modules.resnet import ResNet
from utils.box_utils import match, crop, make_anchors
from modules.swin_transformer import SwinTransformer
from modules.upanets_lite_v6 import UPANets
#import pdb
#import sys
#sys.path.append('C:/Users/User/Hank/Yolact_minimal-master')
from utils import timer_env 
from utils.functions import MovingAverage
from loss import FCOSLoss
from postprocess import FCOSPostprocessor
from fcos import FCOS

from einops import rearrange
#%%
class SPA(nn.Module):
    '''Spatial Pixel Attention'''

    def __init__(self, pixel):
        
        super(SPA, self).__init__()

        self.sp_ffc = nn.Sequential(
            nn.Linear(pixel, pixel)
            )   
        
    def forward(self, x):
        
#        _, p, c = x.shape          
#        x = rearrange(x, 'b p c -> b c p', c=c,p=p)
        x = self.sp_ffc(x)
#        _, c, p = x.shape        
#        out = rearrange(x, 'b c p -> b p c', c=c,p=p)

        return x
    
class PredictionModule(nn.Module):
    def __init__(self, cfg, coef_dim=32):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        self.coef_dim = coef_dim
        
        in_channel = 256

        # self.upfeature = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                       # nn.ReLU(inplace=True))
        self.bbox_layer = nn.Conv2d(in_channel, len(cfg.aspect_ratios) * 4, kernel_size=3, padding=1)
        self.conf_layer = nn.Conv2d(in_channel, len(cfg.aspect_ratios) * self.num_classes, kernel_size=3, padding=1)
        
        self.coef_layer = nn.Sequential(nn.Conv2d(in_channel, len(cfg.aspect_ratios) * self.coef_dim,
                                                      kernel_size=3, padding=1),
                                            nn.Tanh()
                                            )
        

    def forward(self, x):
        # x = self.upfeature(x)
        conf = self.conf_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.num_classes)
        box = self.bbox_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)
        coef = self.coef_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
        
        return conf, box, coef

class ProtoNet(nn.Module):
    def __init__(self, coef_dim):
        super().__init__()
        self.proto1 = nn.Sequential(
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True)
                                    )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.proto2 = nn.Sequential(
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True),                
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, coef_dim, kernel_size=1, stride=1),
                                    nn.ReLU(inplace=True)
                                    )

    def forward(self, x):
        x = self.proto1(x)
        x = self.upsample(x)
        x = self.proto2(x)
        return x


class InvFPN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.lat_layers = nn.ModuleList([nn.Conv2d(x, 256, kernel_size=1) for x in self.in_channels])
        self.pred_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                                        nn.ReLU(inplace=True)) for _ in self.in_channels])

        self.downsample_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                                                              nn.ReLU(inplace=True)),
                                                nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                                                              nn.ReLU(inplace=True))])

        self.upsample_module = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                              nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)])

    def forward(self, backbone_outs):
        p5_1 = self.lat_layers[2](backbone_outs[2])
        p5_upsample = self.upsample_module[1](p5_1)

        p4_1 = self.lat_layers[1](backbone_outs[1]) + p5_upsample
        p4_upsample = self.upsample_module[0](p4_1)

        p3_1 = self.lat_layers[0](backbone_outs[0]) + p4_upsample

        p5 = self.pred_layers[2](p5_1)
        p4 = self.pred_layers[1](p4_1)
        p3 = self.pred_layers[0](p3_1)

        p6 = self.downsample_layers[0](p5)
        p7 = self.downsample_layers[1](p6)

        return p3, p4, p5, p6, p7
    
class FPN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.lat_layers = nn.ModuleList([nn.Conv2d(x, 256, kernel_size=1) for x in self.in_channels])
        self.pred_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                                        nn.ReLU(inplace=True)) for _ in self.in_channels])

        self.downsample_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                                                              nn.ReLU(inplace=True)),
                                                nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                                                              nn.ReLU(inplace=True))])

        self.upsample_module = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                              nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)])

    def forward(self, backbone_outs):
        p5_1 = self.lat_layers[2](backbone_outs[2])
        p5_upsample = self.upsample_module[1](p5_1)

        p4_1 = self.lat_layers[1](backbone_outs[1]) + p5_upsample
        p4_upsample = self.upsample_module[0](p4_1)

        p3_1 = self.lat_layers[0](backbone_outs[0]) + p4_upsample

        p5 = self.pred_layers[2](p5_1)
        p4 = self.pred_layers[1](p4_1)
        p3 = self.pred_layers[0](p3_1)

        p6 = self.downsample_layers[0](p5)
        p7 = self.downsample_layers[1](p6)

        return p3, p4, p5, p6, p7

class Yolact(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        

        self.cfg = cfg
        
        self.coef_dim = 4

#        if cfg.__class__.__name__.startswith('res101'):
#            self.backbone = ResNet(layers=(3, 4, 23, 3))
#            self.fpn = FPN(in_channels=(512, 1024, 2048))
#        elif cfg.__class__.__name__.startswith('res50'):
#            self.backbone = ResNet(layers=(3, 4, 6, 3))
#            self.fpn = FPN(in_channels=(512, 1024, 2048))
#        elif cfg.__class__.__name__.startswith('swin_tiny'):
#            self.backbone = SwinTransformer()
#            self.fpn = FPN(in_channels=(192, 384, 768))

#        self.backbone = ResNet(layers=(3, 4, 6, 3))
#        self.fpn = FPN(in_channels=(512, 1024, 2048))
        
        self.backbone = UPANets(64, 5, 1, 544)
#        self.fcos = FCOS(self.backbone)
#        self.upanet = self.upanet.cuda()
        
        # self.fpn = FPN(in_channels=(256, 512, 1024))
##        self.fpn = FPN(in_channels=(128,256,512))
#        if self.cfg.af == True:
#            self.cfg.og = False
#            
#        if self.cfg.og == True:
        self.proto_net = ProtoNet(coef_dim=self.coef_dim)
        
#        if self.cfg.af == True:
#            self.prediction_layers  = FCOSHead(n_class=self.cfg.num_classes)
#        else
        self.prediction_layers = PredictionModule(cfg, coef_dim=self.coef_dim)

#        self.spa_conf = SPA(cfg.num_classes)
#        self.spa_box = SPA(4)
#        self.spa_coef = SPA(self.coef_dim)
#        self.proto_layer = nn.Conv2d(1984, 256, kernel_size=3, padding=1)
        
        self.anchors = []
        self.num_level_bboxes = []
        fpn_fm_shape = [math.ceil(cfg.img_size / stride) for stride in (8, 16, 32, 64, 128)]
#        fpn_fm_shape = [math.ceil(cfg.img_size / stride) for stride in (8, 16, 32, 64)]
        for i, size in enumerate(fpn_fm_shape):
            # print(i, size)
            anchor = make_anchors(self.cfg, size, size, self.cfg.scales[i])
            # anchor = make_anchors(self.cfg, size, size, self.cfg.scales[0])
            self.anchors += anchor
            # print('anchor size:', len(anchor))
            self.num_level_bboxes.append(int(len(anchor)/4))
        # print('anchor size all:', len(self.anchors))
        self.softmax = torch.nn.Softmax(-1)
#        self.sigmoid = torch.nn.Sigmoid()
#        if self.cfg.og == True:
#            if cfg.mode == 'train':
        # self.semantic_seg_conv = nn.Conv2d(256, cfg.num_classes - 1, kernel_size=1)
#        else:
##            self.semantic_seg_conv = nn.Conv2d(256, cfg.max_detections, kernel_size=1)
#            self.semantic_seg_conv = nn.Conv2d(256, cfg.num_classes - 1, kernel_size=3, padding = 1)
        
        # init weights, backbone weights will be covered later
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight.data)

                if module.bias is not None:
                    module.bias.data.zero_()

    def load_weights(self, weight, cuda):
        if cuda:
            state_dict = torch.load(weight)
        else:
            state_dict = torch.load(weight, map_location='cpu')

        for key in list(state_dict.keys()):
            if self.cfg.mode != 'train' and key.startswith('semantic_seg_conv'):
                del state_dict[key]

        self.load_state_dict(state_dict, strict=True)
        print(f'Model loaded with {weight}.\n')
        print(f'Number of all parameters: {sum([p.numel() for p in self.parameters()])}\n')

    def forward(self, img, box_classes=None, masks_gt=None, img_size=None):
        
#        if self.cfg.upanet == True:
#            outs = self.upanet(img)
#            class_pred = outs[0]
#            box_pred = outs[1]
#            seg_pred = outs[2]
#            
#            if self.training:
#                return self.compute_loss(class_pred, box_pred, 0, 0, seg_pred, box_classes, masks_gt)
#    
#            class_pred = F.softmax(class_pred, -1)
#            
#            return class_pred, box_pred, 0, seg_pred
                
        with timer_env.env('backbone'):
            # outs = self.backbone(img)
            seg_pred, proto_out, class_pred, box_pred, coef_pred = self.backbone(img)
            # sem_out, ins_out, cls_out, box_out, coe_out
            # print('seg:', seg_pred.size())
            # print('proto:', proto_out.size())
            # print('class:', class_pred.size())
            # print('box:', box_pred.size())
            # print('coef:', coef_pred.size())
            # print(coef_pred.size())
            # fpn = self.backbone(img)
            #(sem_out, out_68s, out_34s, out_17s, out_09s, out_05s)
            
#            for i in outs:
#                print(i.shape)
        
        # with timer_env.env('fpn'):
        #     outs = fpn[1:]
            
        with timer_env.env('proto'):
            # proto_out = self.proto_net(outs[0])  # feature map P3
            proto_out = self.proto_net(proto_out)
            # proto_out = self.proto_net(outs[0])
            proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

##        proto_out = self.proto_layer(fpn[0])
##        proto_out = ins_pred.permute(0, 2, 3, 1).contiguous()
        # class_pred, box_pred, coef_pred = [], [], []
        
#         with timer_env.env('pred'):
#             for aa in outs[:]:
# #                print(aa.shape)
#                 class_p, box_p, coef_p = self.prediction_layers(aa)
#                 coef_pred.append(coef_p)
#                 class_pred.append(class_p)
#                 box_pred.append(box_p)
# # ####
#             class_pred = torch.cat(class_pred, dim=1)
#             box_pred = torch.cat(box_pred, dim=1)
#             coef_pred = torch.cat(coef_pred, dim=1)
            
#            class_pred = class_pred + self.spa_conf(class_pred)
#            box_pred = box_pred + self.spa_box(box_pred)
#            coef_pred = coef_pred + self.spa_coef(coef_pred)
        
        with timer_env.env('seg'):
            # seg_pred = fpn[0]
            # seg_pred = self.semantic_seg_conv(outs[0])
            outs = None
            if self.training:
                
                
                # print('class_pred: ', class_pred.shape)
                # print('box_pred: ', box_pred.shape)
                # print('coef_pred: ', coef_pred.shape)
                # print('proto_out: ', proto_out.shape)
                # print('seg_pred: ',seg_pred.shape)
                return self.compute_loss(class_pred, box_pred, coef_pred, proto_out, seg_pred, box_classes, masks_gt, outs)
            
#            class_pred = class_pred * torch.sigmoid(coef_pred)
#            class_pred = self.softmax(class_pred)
            # seg_pred = self.sigmoid(seg_pred)
            
            return class_pred, box_pred, coef_pred, proto_out, seg_pred
    
    def compute_location(self, features):
        
        locations = []
        
        fpn_strides = [8, 16, 32, 64, 128]

        for i, feat in enumerate(features):
            _, _, height, width = feat.shape
            location_per_level = self.compute_location_per_level(
                height, width, fpn_strides[i], feat.device
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
    
    def compute_loss(self, class_p, box_p, coef_p, proto_p, seg_p, box_class, mask_gt, fpn_features):
        # print("-----")
        # print(box_class)
        # print(box_class[0][:])
        # print(box_class[0][:, :-1])
        # print(box_class[0][:, -1])
        # print(box_class[0][:, -1].long())
        
        
        device = class_p.device
        class_gt = [None] * len(box_class)
        batch_size = box_p.size(0)
        
        # print(class_gt)
        # print(batch_size)
        
        # points = self.compute_location(fpn_features)
        points = None
        # print('why????', len(self.anchors))
        if isinstance(self.anchors, list):
            self.anchors = torch.tensor(self.anchors, device=device).reshape(-1, 4)
        # print('after isinstance:', self.anchors.shape)
        num_anchors = self.anchors.shape[0]
        # print('num anchor:', num_anchors)
        # print('anchor shape:', self.anchors.shape)
        

        all_offsets = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float32, device=device)
        conf_gt = torch.zeros((batch_size, num_anchors), dtype=torch.int64, device=device)
        anchor_max_gt = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float32, device=device)
        anchor_max_i = torch.zeros((batch_size, num_anchors), dtype=torch.int64, device=device)

        for i in range(batch_size):
            box_gt = box_class[i][:, :-1]
            class_gt[i] = box_class[i][:, -1].long()
            # all_offsets[i], conf_gt[i], anchor_max_gt[i], anchor_max_i[i] = match(self.cfg, box_gt,
                                                                                  # self.anchors, class_gt[i])

            all_offsets[i], conf_gt[i], anchor_max_gt[i], anchor_max_i[i] = match(self.cfg,
                                                                                  box_gt,
                                                                                  self.anchors,
                                                                                  class_gt[i],
                                                                                  mask_gt[i],
                                                                                  self.num_level_bboxes,
                                                                                  points)
        
        # # print(all_offsets[:])
        # print(conf_gt[:])
        # print(anchor_max_gt[:])
        # print(anchor_max_i[:])

        # all_offsets: the transformed box coordinate offsets of each pair of anchor and gt box
        # conf_gt: the foreground and background labels according to the 'pos_thre' and 'neg_thre',
        #          '0' means background, '>0' means foreground.
        # anchor_max_gt: the corresponding max IoU gt box for each anchor
        # anchor_max_i: the index of the corresponding max IoU gt box for each anchor
        assert (not all_offsets.requires_grad) and (not conf_gt.requires_grad) and \
               (not anchor_max_i.requires_grad), 'Incorrect computation graph, check the grad.'

        # only compute losses from positive samples
        pos_bool = conf_gt > 0
        
#         seg_p_gt =  torch.sigmoid(seg_p)#.gt(0.5).float()
# #        print('seg_p_gt: ', seg_p_gt.shape)
# #        print('randn: ', seg_p_gt[:, :1, :, :].shape)
#         seg_p_gt = torch.cat([torch.zeros(seg_p_gt[:, :1, :, :].shape).cuda(), seg_p_gt], 1)
# #        print('seg_p_gt: ', seg_p_gt.shape)
#         seg_p_gt, _ = seg_p_gt.reshape(seg_p_gt.shape[0], seg_p_gt.shape[1], -1).max(2, True)
# #        print('seg_p_gt: ', seg_p_gt.shape)
#         seg_p_gt = seg_p_gt.permute(0, 2, 1)
# #        print('seg_p_gt: ', seg_p_gt.shape)
#         # cs = class_p * seg_p_gt
#         cs = self.softmax(class_p) * seg_p_gt
        
# #        cls_p_gt = torch.sigmoid(class_p)
# #        print('cls_p_gt: ', cls_p_gt.shape)
# #        print('seg_p_gt: ', seg_p_gt.shape)
# #        cs = cls_p_gt * seg_p_gt
#         cs, _ = torch.max(cs, dim=2, keepdim=True)
#         keep = (cs > self.cfg.nms_score_thre)
# #        keep = (cs > cs.mean(1, True))
# #        print('keep: ',keep.shape)
# #        import numpy as np
# #        print('keep unique: ', np.unique(keep.detach().cpu().numpy()))
        keep = None
        
        loss_c = self.category_loss(class_p, conf_gt, pos_bool, keep)
        loss_b = self.box_loss(box_p, all_offsets, pos_bool, keep)
        if self.cfg.og == True:
            loss_m = self.lincomb_mask_loss(pos_bool, anchor_max_i, coef_p, proto_p, mask_gt, anchor_max_gt, keep)
            # loss_m = self.lincomb_mask_loss_og(pos_bool, anchor_max_i, coef_p, proto_p, mask_gt, anchor_max_gt)
        else:
            loss_m = 0
        loss_s = self.semantic_seg_loss(seg_p, mask_gt, class_gt)
        return loss_c, loss_b, loss_m, loss_s

    def category_loss(self, class_p, conf_gt, pos_bool, keep, np_ratio=3):
        # Compute max conf across batch for hard negative mining
        if keep != None:
            class_p = class_p * keep
            
            
        # print('pos:', pos_bool.size())  
        
        batch_conf = class_p.reshape(-1, self.cfg.num_classes)
        # print('batch conf:', batch_conf.size())
        
        batch_conf_max = batch_conf.max()
        # print('batch conf max:', batch_conf_max)
        mark = torch.log(torch.sum(torch.exp(batch_conf - batch_conf_max), 1)) + batch_conf_max - batch_conf[:, 0]
        # print('mark:', mark.size())
        # Hard Negative Mining
        mark = mark.reshape(class_p.size(0), -1)
        # print('mark2:', mark.size())
        mark[pos_bool] = 0  # filter out pos boxes
        mark[conf_gt < 0] = 0  # filter out neutrals (conf_gt = -1)

        _, idx = mark.sort(1, descending=True)
        _, idx_rank = idx.sort(1)

        num_pos = pos_bool.long().sum(1, keepdim=True)
        num_neg = torch.clamp(np_ratio * num_pos, max=pos_bool.size(1) - 1)
        neg_bool = idx_rank < num_neg.expand_as(idx_rank)

        # Just in case there aren't enough negatives, don't start using positives as negatives
        neg_bool[pos_bool] = 0
        neg_bool[conf_gt < 0] = 0  # Filter out neutrals

        # Confidence Loss Including Positive and Negative Examples
        class_p_mined = class_p[(pos_bool + neg_bool)].reshape(-1, self.cfg.num_classes)
        class_gt_mined = conf_gt[(pos_bool + neg_bool)]

        return self.cfg.conf_alpha * F.cross_entropy(class_p_mined, class_gt_mined, reduction='sum') / num_pos.sum()

    def box_loss(self, box_p, all_offsets, pos_bool, keep):
        
        if keep != None:
#            print('box_p: ', box_p.shape)
            box_p = box_p * keep
        
        num_pos = pos_bool.sum()
        pos_box_p = box_p[pos_bool, :]
        pos_offsets = all_offsets[pos_bool, :]
        
        if self.cfg.og == True:
            return self.cfg.bbox_alpha * F.smooth_l1_loss(pos_box_p, pos_offsets, reduction='sum') / num_pos
        
        return self.cfg.bbox_alpha * F.smooth_l1_loss(pos_box_p, pos_offsets, reduction='sum') / num_pos
        
    def lincomb_mask_loss(self, pos_bool, anchor_max_i, coef_p, proto_p, mask_gt, anchor_max_gt, keep):
        
        if keep != None:
            coef_p = coef_p * keep
        
        proto_h, proto_w = proto_p.shape[1:3]
        total_pos_num = pos_bool.sum()
        loss_m = 0
#        coef_p = torch.tanh(coef_p)
        for i in range(coef_p.size(0)):
            # downsample the gt mask to the size of 'proto_p'
            downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()
            # binarize the gt mask because of the downsample operation
            downsampled_masks = downsampled_masks.gt(0.5).float()

            pos_anchor_i = anchor_max_i[i][pos_bool[i]]
            pos_anchor_box = anchor_max_gt[i][pos_bool[i]]
            pos_coef = coef_p[i][pos_bool[i]]

            if pos_anchor_i.size(0) == 0:
                continue

            # If exceeds the number of masks for training, select a random subset
            old_num_pos = pos_coef.size(0)
            if old_num_pos > self.cfg.masks_to_train:
                perm = torch.randperm(pos_coef.size(0))
                select = perm[:self.cfg.masks_to_train]
                pos_coef = pos_coef[select]
                pos_anchor_i = pos_anchor_i[select]
                pos_anchor_box = pos_anchor_box[select]

            num_pos = pos_coef.size(0)

            pos_mask_gt = downsampled_masks[:, :, pos_anchor_i]

            # mask assembly by linear combination
            # @ means dot product
#            mask_p = torch.sigmoid(proto_p[i] @ pos_coef.t())
            mask_p = proto_p[i] @ pos_coef.t()
            mask_p = crop(mask_p, pos_anchor_box)  # pos_anchor_box.shape: (num_pos, 4)
            # TODO: grad out of gt box is 0, should it be modified?
            # TODO: need an upsample before computing loss?
#            print('torch.clamp(mask_p, 0, 1): ', torch.clamp(mask_p, 0, 1).shape)
#            print('pos_mask_gt: ',pos_mask_gt.shape)
            mask_loss = F.binary_cross_entropy_with_logits(mask_p, pos_mask_gt, reduction='none')
#            mask_loss = F.binary_cross_entropy(torch.clamp(mask_p, min=0., max=1.), pos_mask_gt, reduction='none')
#            mask_loss = F.binary_cross_entropy(mask_p, pos_mask_gt, reduction='none')
#            mask_loss = -pos_mask_gt*torch.log(mask_p) - (1-pos_mask_gt) * torch.log(1-mask_p)

            # Normalize the mask loss to emulate roi pooling's effect on loss.
            anchor_area = (pos_anchor_box[:, 2] - pos_anchor_box[:, 0]) * (pos_anchor_box[:, 3] - pos_anchor_box[:, 1])
            mask_loss = mask_loss.sum(dim=(0, 1)) / anchor_area
#            mask_loss = mask_loss / anchor_area

            if old_num_pos > num_pos:
                mask_loss *= old_num_pos / num_pos

            loss_m += torch.sum(mask_loss)
#            loss_m += mask_loss
            
            # self.cfg.mask_alpha = 8.0
            
        return self.cfg.mask_alpha * loss_m / proto_h / proto_w / total_pos_num
#        return self.cfg.mask_alpha * loss_m / total_pos_num
    def lincomb_mask_loss_og(self, pos_bool, anchor_max_i, coef_p, proto_p, mask_gt, anchor_max_gt):
        proto_h, proto_w = proto_p.shape[1:3]
        total_pos_num = pos_bool.sum()
        loss_m = 0
        for i in range(coef_p.size(0)):
            # downsample the gt mask to the size of 'proto_p'
            downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()
            # binarize the gt mask because of the downsample operation
            downsampled_masks = downsampled_masks.gt(0.5).float()

            pos_anchor_i = anchor_max_i[i][pos_bool[i]]
            pos_anchor_box = anchor_max_gt[i][pos_bool[i]]
            pos_coef = coef_p[i][pos_bool[i]]

            if pos_anchor_i.size(0) == 0:
                continue

            # If exceeds the number of masks for training, select a random subset
            old_num_pos = pos_coef.size(0)
            if old_num_pos > self.cfg.masks_to_train:
                perm = torch.randperm(pos_coef.size(0))
                select = perm[:self.cfg.masks_to_train]
                pos_coef = pos_coef[select]
                pos_anchor_i = pos_anchor_i[select]
                pos_anchor_box = pos_anchor_box[select]

            num_pos = pos_coef.size(0)

            pos_mask_gt = downsampled_masks[:, :, pos_anchor_i]

            # mask assembly by linear combination
            # @ means dot product
            mask_p = torch.sigmoid(proto_p[i] @ pos_coef.t())
            mask_p = crop(mask_p, pos_anchor_box)  # pos_anchor_box.shape: (num_pos, 4)
            # TODO: grad out of gt box is 0, should it be modified?
            # TODO: need an upsample before computing loss?
#            print('torch.clamp(mask_p, 0, 1): ', torch.clamp(mask_p, 0, 1).shape)
#            print('pos_mask_gt: ',pos_mask_gt.shape)
#            mask_loss = F.binary_cross_entropy_with_logits(torch.clamp(mask_p, min=0., max=1.), pos_mask_gt, reduction='none')
            mask_loss = F.binary_cross_entropy(torch.clamp(mask_p, min=0., max=1.), pos_mask_gt, reduction='none')
            # mask_loss = -pos_mask_gt*torch.log(mask_p) - (1-pos_mask_gt) * torch.log(1-mask_p)

            # Normalize the mask loss to emulate roi pooling's effect on loss.
            anchor_area = (pos_anchor_box[:, 2] - pos_anchor_box[:, 0]) * (pos_anchor_box[:, 3] - pos_anchor_box[:, 1])
            mask_loss = mask_loss.sum(dim=(0, 1)) / anchor_area
#            mask_loss = mask_loss / anchor_area

            if old_num_pos > num_pos:
                mask_loss *= old_num_pos / num_pos

            loss_m += torch.sum(mask_loss)

        return self.cfg.mask_alpha * loss_m / proto_h / proto_w / total_pos_num

    def semantic_seg_loss(self, segmentation_p, mask_gt, class_gt):
        # Note classes here exclude the background class, so num_classes = cfg.num_classes - 1
        batch_size, num_classes, mask_h, mask_w = segmentation_p.size()
        loss_s = 0

        for i in range(batch_size):
            cur_segment = segmentation_p[i]
            cur_class_gt = class_gt[i]

            downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0), (mask_h, mask_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
#            downsampled_masks = mask_gt[i]
            downsampled_masks = downsampled_masks.gt(0.5).float()

            # Construct Semantic Segmentation
            segment_gt = torch.zeros_like(cur_segment, requires_grad=False)
            for j in range(downsampled_masks.size(0)):
#                print('segment_gt: ', segment_gt[cur_class_gt[j]].shape)
#                print('downsample_masks: ', downsampled_masks[j].shape)
                segment_gt[cur_class_gt[j]] = torch.max(segment_gt[cur_class_gt[j]], downsampled_masks[j])

            loss_s += F.binary_cross_entropy_with_logits(cur_segment, segment_gt, reduction='sum')
        
        if self.cfg.og == True:
            return self.cfg.semantic_alpha * loss_s / mask_h / mask_w / batch_size
#            return  self.cfg.semantic_alpha * loss_s / batch_size
        
        return (self.cfg.mask_alpha + self.cfg.semantic_alpha) * loss_s / mask_h / mask_w / batch_size
#        return loss_s / mask_h / mask_w / batch_size

##%%
#cfg.og =False
##net = Yolact(cfg)
#net = UPANets(16, 21, 1, 544)
#net.eval()
##net.eval()
#net = net.cuda()
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
#x = torch.zeros((1, 3, 544, 544)).cuda()
#y = net(x)
#avg = MovingAverage()
#timer.reset()
#with timer.env('everything else'):
#    net(x)
#avg.add(timer.total_time())
#print('\033[2J') # Moves console cursor to 0,0
#timer.print_stats()
#print('Avg fps: %.2f\tAvg ms: %.2f         ' % (1/avg.get_avg(), avg.get_avg()*1000))    
#%%
##cfg.og =True
##net = construct_backbone(cfg.backbone)
##net = ResNet(layers=(3, 4, 6, 3))
#net = Yolact(cfg)
##net = UPANets(cfg, 16, 21, 1, 544)
##net = UPANets(64, 21, 1, 544)
##net = Yolact()
##net.train()
##backbone = vovnet39()
##net = FCOS(args, backbone)
#net.eval()
#net = net.cuda()
##torch.set_default_tensor_type('torch.cuda.FloatTensor')
#x = torch.randn((1, 3, 544, 544)).cuda()
#
##%%
#import time
#s = time.time()
#y = net(x)
#d = time.time() - s
#f = 1/d
##%%
#y = net(x)
##net(x)
#avg = MovingAverage()
##net.eval()
#timer_env.reset()
#try:
#    while True:
#        timer_env.reset()
#        with timer_env.env('everything else'):
#            net(x)
#        avg.add(timer_env.total_time())
#        print('\033[2J') # Moves console cursor to 0,0
#        timer_env.print_stats()
#        print('Avg fps: %.2f\tAvg ms: %.2f         ' % (1/avg.get_avg(), avg.get_avg()*1000))
#except KeyboardInterrupt:
#    pass
#with timer_env.env('everything else'):
#    net(x)
#avg.add(timer_env.total_time())
#print('\033[2J') # Moves console cursor to 0,0
#timer_env.print_stats()
#print('Avg fps: %.2f\tAvg ms: %.2f         ' % (1/avg.get_avg(), avg.get_avg()*1000))