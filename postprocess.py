import torch
from torch import nn
import torch.nn.functional as F
from boxlist import BoxList, boxlist_nms, remove_small_box, cat_boxlist


class FCOSPostprocessor(nn.Module):
    def __init__(self, threshold, top_n, nms_threshold, post_top_n, min_size, n_class):
        super().__init__()

        self.threshold = threshold
        self.top_n = top_n
        self.nms_threshold = nms_threshold
        self.post_top_n = post_top_n
        self.min_size = min_size
        self.n_class = n_class

    def forward_single_feature_map(
        self, location, cls_pred, box_pred, center_pred, image_sizes
    ):
        
        batch, channel, height, width = cls_pred.shape

        cls_pred = cls_pred.view(batch, channel, height, width).permute(0, 2, 3, 1)
        cls_pred = cls_pred.reshape(batch, -1, channel).sigmoid()
#        cur_loc = cls_pred.shape[1]
        
        box_pred = box_pred.view(batch, 4, height, width).permute(0, 2, 3, 1)
        box_pred = box_pred.reshape(batch, -1, 4)

        center_pred = center_pred.view(batch, 1, height, width).permute(0, 2, 3, 1)
        center_pred = center_pred.reshape(batch, -1).sigmoid()

        candid_ids = cls_pred > self.threshold
#        top_ns = candid_ids.view(batch, -1).sum(1)
        top_ns = candid_ids.reshape(batch, -1).sum(1)
        top_ns = top_ns.clamp(max=self.top_n)

        cls_pred = cls_pred * center_pred[:, :, None]

        results = []

        for i in range(batch):
            cls_p = cls_pred[i]
            candid_id = candid_ids[i]
            cls_p = cls_p[candid_id]
            candid_nonzero = candid_id.nonzero()
            box_loc = candid_nonzero[:, 0]
            class_id = candid_nonzero[:, 1]# + 1

            box_p = box_pred[i]
            box_p = box_p[box_loc]
            loc = location[box_loc]

            top_n = top_ns[i]

            if candid_id.sum().item() > top_n.item():
                cls_p, top_k_id = cls_p.topk(top_n, sorted=False)
                class_id = class_id[top_k_id]
                box_p = box_p[top_k_id]
                loc = loc[top_k_id]

            detections = torch.stack(
                [
                    loc[:, 0] - box_p[:, 0],
                    loc[:, 1] - box_p[:, 1],
                    loc[:, 0] + box_p[:, 2],
                    loc[:, 1] + box_p[:, 3],
                ],
                1,
            )

#            height, width = image_sizes[i]
            height, width = image_sizes

            boxlist = BoxList(detections, (int(width), int(height)), mode='xyxy')
            boxlist.fields['labels'] = class_id
            boxlist.fields['scores'] = torch.sqrt(cls_p)
            boxlist = boxlist.clip(remove_empty=False)
            boxlist = remove_small_box(boxlist, self.min_size)

            results.append(boxlist)
            
#        out_loc = [b + self.loc for b in box_loc]
#        self.loc = self.loc + cur_loc
        
        return results, box_loc

    def forward(self, location, cls_pred, box_pred, center_pred, seg_pred, ins_pred, can_pred, image_sizes):
        boxes = []
        locs = []
#        og_locs = []
#        import numpy as np
        acc_loc = 0
        for loc, cls_p, box_p, center_p in zip(
            location, cls_pred, box_pred, center_pred
        ):
            box, loc = self.forward_single_feature_map(loc, cls_p, box_p, center_p, image_sizes)
            
            if loc != []:
                for l in loc:
                    locs.append(l + acc_loc)
            boxes.append(box)
                
#            og_locs.append(loc)
            acc_loc = acc_loc + cls_p.shape[2]**2
        
        locs = torch.tensor(locs)
        boxlists = list(zip(*boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists, loclists = self.select_over_scales(boxlists, locs)

#        return boxlists
        box_p = boxlists[0]#.convert('xywh')
        score_p, class_p, box_p = boxlists[0].fields['scores'], boxlists[0].fields['labels'], box_p.box.int()
#        print('seg_pred: ',seg_pred.shape)
#        print('ins_pred: ', ins_pred.squeeze(0)[loclists.long(), :].shape)
#        print('loclists: ',len(loclists))
#        import numpy as np
#        print('unique: ', len(np.unique(loclists)))
        seg_pred = can_pred
        seg_p = seg_pred.squeeze(0).permute(1,2,0) @ ins_pred.squeeze(0)[loclists.long(), :].t()
        
#        print('seg_p: ', seg_p.shape)
#        print('image_sizes: ', image_sizes)
        seg_p = F.interpolate(seg_p.permute(2,0,1).unsqueeze(0), image_sizes, mode='bilinear', align_corners=False).squeeze(0)
        seg_p = torch.sigmoid(seg_p)
#        seg_p = seg_p > 0.5
#        print(np.unique(seg_p.detach().numpy()))
        seg_p.gt_(0.5)  # Binarize the masks because of interpolation.
#        
#        print(np.unique(seg_p.detach().numpy()))
#        print('seg_p 0.5: ',seg_p.shape)
        seg_p = seg_p[:, 0: image_sizes[0], :] if image_sizes[0] < image_sizes[1] else seg_p[:, :, 0: image_sizes[1]]
        
        return score_p, class_p, box_p, seg_p

    def select_over_scales(self, boxlists, locs):
        results = []
        loc_results = []
        for boxlist in boxlists:
            scores = boxlist.fields['scores']
            labels = boxlist.fields['labels']
            box = boxlist.box

            result = []
            loc_result = []
            for j in range(1, self.n_class):
                id = (labels == j).nonzero().view(-1)
                score_j = scores[id]
                box_j = box[id, :].view(-1, 4)
                loc_j = locs[id]
                box_by_class = BoxList(box_j, boxlist.size, mode='xyxy')
                box_by_class.fields['scores'] = score_j
                box_by_class, loc_nms = boxlist_nms(box_by_class, score_j, loc_j, self.nms_threshold)
                n_label = len(box_by_class)
                box_by_class.fields['labels'] = torch.full(
                    (n_label,), j, dtype=torch.int64, device=scores.device
                )
                result.append(box_by_class)
                loc_result.append(loc_nms)
            result = cat_boxlist(result)
            loc_result = torch.cat(loc_result)
            n_detection = len(result)

            if n_detection > self.post_top_n > 0:
                scores = result.fields['scores']
                img_threshold, _ = torch.kthvalue(
                    scores.cpu(), n_detection - self.post_top_n + 1
                )
                keep = scores >= img_threshold.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
                loc_result = loc_result[keep]

            results.append(result)
            loc_results.append(loc_results)

        return results, loc_result
