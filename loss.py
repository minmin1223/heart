import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from boxlist import BoxList

INF = 100000000

def bbox2bbox(x,y,w,h):
    
    x,y,w,h = x*544,y*544,w*544,h*544 
    device = x.device if hasattr(x, 'device') else 'cpu'
    x1, y1 = x, y
    x2, y2 = x+w, y+h
    
    return torch.as_tensor([(x1, y1, x2, y2)], dtype=torch.float32, device=device)

def sanitize_coordinates(_x1, _x2, img_size, padding=0):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size

    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1 - padding, min=0)
    x2 = torch.clamp(x2 + padding, max=img_size)

    return x1, x2

def crop(masks, boxes, padding=1):
    # masks: [136, 136, n], boxes: [n, 4]
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    # 136, 136, n
    h, w, n = masks.size()
    # n, n
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding)
    # n, n
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding)
    
    # [136, 136, 9]
    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
    # [136, 136, 9]
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)

    # [136, 136, 9] >= [1, 1, 9]
    masks_left = rows >= x1.view(1, 1, -1)
    masks_right = rows < x2.view(1, 1, -1)
    masks_up = cols >= y1.view(1, 1, -1)
    masks_down = cols < y2.view(1, 1, -1)

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask.float()

class IOULoss(nn.Module):
    def __init__(self, loc_loss_type):
        super().__init__()

        self.loc_loss_type = loc_loss_type

    def forward(self, out, target, weight=None):
        pred_left, pred_top, pred_right, pred_bottom = out.unbind(1)
        target_left, target_top, target_right, target_bottom = target.unbind(1)

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(
            pred_right, target_right
        )
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(
            pred_top, target_top
        )

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        ious = (area_intersect + 1) / (area_union + 1)

        if self.loc_loss_type == 'iou':
            loss = -torch.log(ious)

        elif self.loc_loss_type == 'giou':
            g_w_intersect = torch.max(pred_left, target_left) + torch.max(
                pred_right, target_right
            )
            g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(
                pred_top, target_top
            )
            g_intersect = g_w_intersect * g_h_intersect + 1e-7
            gious = ious - (g_intersect - area_union) / g_intersect

            loss = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (loss * weight).sum() / weight.sum()

        else:
            return loss.mean()


def clip_sigmoid(input):
    out = torch.clamp(torch.sigmoid(input), min=1e-4, max=1 - 1e-4)

    return out


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, out, target):
        n_class = out.shape[1]
        class_ids = torch.arange(
            1, n_class + 1, dtype=target.dtype, device=target.device
        ).unsqueeze(0)

        t = target.unsqueeze(1)
        p = torch.sigmoid(out)

        gamma = self.gamma
        alpha = self.alpha

        term1 = (1 - p) ** gamma * torch.log(p)
        term2 = p ** gamma * torch.log(1 - p)

        # print(term1.sum(), term2.sum())

        loss = (
            -(t == class_ids).float() * alpha * term1
            - ((t != class_ids) * (t >= 0)).float() * (1 - alpha) * term2
        )

        return loss.sum()


class FCOSLoss(nn.Module):
    def __init__(
        self, sizes, gamma, alpha, iou_loss_type, center_sample, fpn_strides, pos_radius
    ):
        super().__init__()

        self.sizes = sizes

        self.cls_loss = SigmoidFocalLoss(gamma, alpha)
        self.box_loss = IOULoss(iou_loss_type)
        self.center_loss = nn.BCEWithLogitsLoss()

        self.center_sample = center_sample
        self.strides = fpn_strides
        self.radius = pos_radius

    def prepare_target(self, points, targets, masks):
        ex_size_of_interest = []

        for i, point_per_level in enumerate(points):
            size_of_interest_per_level = point_per_level.new_tensor(self.sizes[i])
            ex_size_of_interest.append(
                size_of_interest_per_level[None].expand(len(point_per_level), -1)
            )

        ex_size_of_interest = torch.cat(ex_size_of_interest, 0)
        n_point_per_level = [len(point_per_level) for point_per_level in points]
        point_all = torch.cat(points, dim=0)
        label, box_target = self.compute_target_for_location(
            point_all, targets, ex_size_of_interest, n_point_per_level
        )
        
        ins_masks = []
        seg_masks = []
        crop_boxs = []
        for i, l in enumerate(label):
            label_type = np.unique(l.cpu().detach().numpy())
            for t in range(len(label_type)):
                if t == 0.:
#                    li=l[l!=t]
                    lii = l[l!=t]
                    seg_masks.append(l!=t)
                    
                else:
                    lii[lii==label_type[t]] = t-1
            m = masks[i].permute(1,2,0).contiguous().gt(0.5).float()
            ins_masks.append(m[:, :, lii.long()])
#            crop_boxs.append(targets[i][:,:-1].permute(1,0)[:, lii.long()].permute(1,0))
#            break
            
            
        for i in range(len(label)):
            label[i] = torch.split(label[i], n_point_per_level, 0)
            box_target[i] = torch.split(box_target[i], n_point_per_level, 0)
                    
        label_level_first = []
        box_target_level_first = []

        for level in range(len(points)):
            label_level_first.append(
                torch.cat([label_per_img[level] for label_per_img in label], 0)
            )
            box_target_level_first.append(
                torch.cat(
                    [box_target_per_img[level] for box_target_per_img in box_target], 0
                )
            )

        return label_level_first, box_target_level_first, ins_masks, seg_masks, crop_boxs

    def get_sample_region(self, gt, strides, n_point_per_level, xs, ys, radius=1):
        n_gt = gt.shape[0]
        n_loc = len(xs)
        gt = gt[None].expand(n_loc, n_gt, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2

        if center_x[..., 0].sum() == 0:
            return xs.new_zeros(xs.shape, dtype=torch.uint8)

        begin = 0

        center_gt = gt.new_zeros(gt.shape)

        for level, n_p in enumerate(n_point_per_level):
            end = begin + n_p
            stride = strides[level] * radius

            x_min = center_x[begin:end] - stride
            y_min = center_y[begin:end] - stride
            x_max = center_x[begin:end] + stride
            y_max = center_y[begin:end] + stride

            center_gt[begin:end, :, 0] = torch.where(
                x_min > gt[begin:end, :, 0], x_min, gt[begin:end, :, 0]
            )
            center_gt[begin:end, :, 1] = torch.where(
                y_min > gt[begin:end, :, 1], y_min, gt[begin:end, :, 1]
            )
            center_gt[begin:end, :, 2] = torch.where(
                x_max > gt[begin:end, :, 2], gt[begin:end, :, 2], x_max
            )
            center_gt[begin:end, :, 3] = torch.where(
                y_max > gt[begin:end, :, 3], gt[begin:end, :, 3], y_max
            )

            begin = end

        left = xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - xs[:, None]
        top = ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - ys[:, None]

        center_bbox = torch.stack((left, top, right, bottom), -1)
        is_in_boxes = center_bbox.min(-1)[0] > 0

        return is_in_boxes

    def compute_target_for_location(
        self, locations, targets, sizes_of_interest, n_point_per_level
    ):
        labels = []
        box_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for i in range(len(targets)):
            
            '''og'''
#            targets_per_img = targets[i]
#            assert targets_per_img.mode == 'xyxy'
#            bboxes = targets_per_img.box
#            labels_per_img = targets_per_img.fields['labels']
#            area = targets_per_img.area()
            '''og'''
            
            '''modify'''
#            bboxes_gt_list = []
#            for j in range(len(targets[i][:, :-1][:])):
#                bboxes_gt_list.append(bbox2bbox(*targets[i][:, :-1][j]))  
#            
#            bboxes = torch.cat(bboxes_gt_list, 0)  
#            labels_per_img = targets[i][:, -1] + 1
#            area = 544*(targets[i][:, -3][:] * targets[i][:, -2][:])
                        
            bboxes = BoxList(targets[i][:, :-1], (544, 544), mode='xyxy')
            area = bboxes.area()*544
            bboxes = bboxes.box*544
            labels_per_img = targets[i][:, -1] + 1
            '''modify'''
            
            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]

            box_targets_per_img = torch.stack([l, t, r, b], 2)

            if self.center_sample:
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, n_point_per_level, xs, ys, radius=self.radius
                )

            else:
                is_in_boxes = box_targets_per_img.min(2)[0] > 0

            max_box_targets_per_img = box_targets_per_img.max(2)[0]

            is_cared_in_level = (
                max_box_targets_per_img >= sizes_of_interest[:, [0]]
            ) & (max_box_targets_per_img <= sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_level == 0] = INF

            locations_to_min_area, locations_to_gt_id = locations_to_gt_area.min(1)

            box_targets_per_img = box_targets_per_img[
                range(len(locations)), locations_to_gt_id
            ]
            labels_per_img = labels_per_img[locations_to_gt_id]
            labels_per_img[locations_to_min_area == INF] = 0

            labels.append(labels_per_img)
            box_targets.append(box_targets_per_img)

        return labels, box_targets

    def compute_centerness_targets(self, box_targets):
        left_right = box_targets[:, [0, 2]]
        top_bottom = box_targets[:, [1, 3]]
        centerness = (left_right.min(-1)[0] / left_right.max(-1)[0]) * (
            top_bottom.min(-1)[0] / top_bottom.max(-1)[0]
        )

        return torch.sqrt(centerness)

    def semantic_seg_loss(self, segmentation_p, mask_gt, class_gt):
        # Note classes here exclude the background class, so num_classes = cfg.num_classes - 1
        batch_size, num_classes, mask_h, mask_w = segmentation_p.size()
        _, h, w = mask_gt[0].shape
        loss_s = 0

        for i in range(batch_size):
            cur_segment = segmentation_p[i]
            cur_class_gt = class_gt[i]

#            downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0), (mask_h, mask_w), mode='bilinear',
#                                              align_corners=False).squeeze(0)
#            downsampled_masks = downsampled_masks.gt(0.5).float()
            downsampled_masks = mask_gt[i].gt(0.5).float()
            cur_segment = F.interpolate(cur_segment.unsqueeze(0), (h, w), mode='bilinear', align_corners=False).squeeze(0)

            # Construct Semantic Segmentation
            segment_gt = torch.zeros_like(cur_segment, requires_grad=False)
#            for j in range(downsampled_masks.size(0)):
            for j in range(mask_gt[i].size(0)):
#                print('segment_gt: ', segment_gt[cur_class_gt[j]].shape)
#                print('downsample_masks: ', downsampled_masks[j].shape)
#                print('segment_gt[cur_class_gt[j]]:',segment_gt[cur_class_gt[j]].shape)
#                print('downsampled_masks[j]:',downsampled_masks[j].shape)
                segment_gt[cur_class_gt[j]] = torch.max(segment_gt[cur_class_gt[j]], downsampled_masks[j])
#                segment_gt[cur_class_gt[j]] = torch.max(segment_gt[cur_class_gt[j]], mask_gt[i][j])

            loss_s = loss_s + F.binary_cross_entropy_with_logits(cur_segment, segment_gt, reduction='sum')
#            loss_s = loss_s + F.binary_cross_entropy_with_logits(cur_segment, segment_gt, reduction='mean')
        
        return loss_s / 544 / 544 / batch_size
#        return loss_s / batch_size
    
    def instance_seg_loss(self, ins_pred_t, seg_pred, ins_masks):
        
        loss = 0
        batch_size = len(ins_pred_t)
        
        h, w, _ = ins_masks[0].shape
        nums = 0
        for i in range(batch_size):
            if ins_pred_t[i].t().shape[1] != 0:
                nums = nums + ins_pred_t[i].t().shape[1]
                ins_pred_masks = seg_pred[i].permute(1,2,0) @ ins_pred_t[i].t()
                ins_pred_masks = F.interpolate(ins_pred_masks.permute(2,0,1).unsqueeze(0), (h, w), mode='bilinear', align_corners=False).squeeze(0)
#                i_loss = F.binary_cross_entropy_with_logits(ins_pred_masks.permute(1,2,0), ins_masks[i], reduction='mean')
                i_loss = F.binary_cross_entropy_with_logits(ins_pred_masks, ins_masks[i].permute(2,0,1), reduction='sum')
#                i_loss = F.binary_cross_entropy_with_logits(ins_pred_masks, ins_masks[i].permute(2,0,1), reduction='mean')
                loss = loss + i_loss
            else:
                batch_size = batch_size - 1
        
        if batch_size == 0:
            return 0
        else:
            return loss / 544 / 544 / batch_size / nums
#            return loss / batch_size
    
    def forward(self, locations, cls_pred, box_pred, center_pred, seg_pred, ins_pred, can_pred, targets, masks):
        batch = cls_pred[0].shape[0]
        n_class = cls_pred[0].shape[1]

        labels, box_targets, ins_masks, seg_masks, crop_boxs = self.prepare_target(locations, targets, masks)

        cls_flat = []
        box_flat = []
        center_flat = []
        

        labels_flat = []
        box_targets_flat = []
#        pos_pixels = []

        for i in range(len(labels)):
            cls_flat.append(cls_pred[i].permute(0, 2, 3, 1).reshape(-1, n_class))
            box_flat.append(box_pred[i].permute(0, 2, 3, 1).reshape(-1, 4))
            center_flat.append(center_pred[i].permute(0, 2, 3, 1).reshape(-1))

            labels_flat.append(labels[i].reshape(-1))
            box_targets_flat.append(box_targets[i].reshape(-1, 4))

        cls_flat = torch.cat(cls_flat, 0)
        box_flat = torch.cat(box_flat, 0)
        center_flat = torch.cat(center_flat, 0)

        labels_flat = torch.cat(labels_flat, 0)
        box_targets_flat = torch.cat(box_targets_flat, 0)

        pos_id = torch.nonzero(labels_flat > 0).squeeze(1)

        cls_loss = self.cls_loss(cls_flat, labels_flat.int()) / (pos_id.numel() + batch)

        box_flat = box_flat[pos_id]
        center_flat = center_flat[pos_id]

        box_targets_flat = box_targets_flat[pos_id]
        
        ins_pred_t = []
        class_gt = [None] * len(targets)
        batch_size = len(targets)
        
        for i in range(batch_size):
            class_gt[i] = targets[i][:, -1].long() 
            ins_pred_t.append(ins_pred[i][seg_masks[i]])
        
#        print('ins_pred_t: ',len(ins_pred_t))
        ins_loss = self.instance_seg_loss(ins_pred_t, can_pred, ins_masks)
        seg_loss = self.semantic_seg_loss(seg_pred, masks, class_gt)
                    

        if pos_id.numel() > 0:
            center_targets = self.compute_centerness_targets(box_targets_flat)

            box_loss = self.box_loss(box_flat, box_targets_flat, center_targets)
            center_loss = self.center_loss(center_flat, center_targets)

        else:
            box_loss = box_flat.sum()
            center_loss = center_flat.sum()

        return cls_loss, box_loss, center_loss, seg_loss, ins_loss
