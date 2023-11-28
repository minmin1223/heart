#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

#import sys
#sys.path.append('/Users/hank/Downloads/Yolact_minimal-master/Yolact_minimal-master/utils')
#sys.path.append('/Users/hank/Downloads/Yolact_minimal-master/Yolact_minimal-master')
#from box_utils import match, crop, make_anchors
#from modules.swin_transformer import SwinTransformer
#import pdb
#import timer_env as timer
#from utils.functions import MovingAverage
class upa_block(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1, cat=False, same=False, w=1, l=2, up=False):
        
        super(upa_block, self).__init__()
        
        self.cat = cat
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        self.same = same
        self.cnn = nn.Sequential(
            nn.Conv2d(in_planes, int(planes * w), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(planes * w)),
            nn.ReLU(),
            nn.Conv2d(int(planes * w), planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
            )
        if l == 1:
            w = 1
            self.cnn = nn.Sequential(
                nn.Conv2d(in_planes, int(planes * w), kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(int(planes * w)),
                nn.ReLU(),
                )
        
        self.att = CPA(in_planes, planes, stride, same=same, up=up)
        
    def forward(self, x):
#        print('in_planes:', self.in_planes)
#        print('planes:', self.planes)
        out = self.cnn(x)
        out = self.att(x, out)

        if self.cat == True:
            out = torch.cat([x, out], 1)
            
        return out

class CPA(nn.Module):
    '''Channel Pixel Attention'''
    
#      *same=False:
#       This scenario can be easily embedded after any CNNs, if size is same.
#        x (OG) ---------------
#        |                    |
#        sc_x (from CNNs)     CPA(x)
#        |                    |
#        out + <---------------
#        
#      *same=True:
#       This can be embedded after the CNNs where the size are different.
#        x (OG) ---------------
#        |                    |
#        sc_x (from CNNs)     |
#        |                    CPA(x)
#        CPA(sc_x)            |
#        |                    |
#        out + <---------------
#           
#      *sc_x=False
#       This operation can be seen a channel embedding with CPA
#       EX: x (3, 32, 32) => (16, 32, 32)
#        x (OG) 
#        |      
#        CPA(x)
#        |    
#        out 
    
    def __init__(self, in_dim, dim, stride=1, same=False, up=False, sc_x=True, final=False):
        
        super(CPA, self).__init__()
            
        self.dim = dim
        self.stride = stride
        self.same = same
        self.sc_x = sc_x
        self.final = final
        
        self.cp_ffc = nn.Linear(in_dim, dim)
        self.bn = nn.BatchNorm2d(dim)
        self.up = up

        if self.stride == 2 or self.same == True:
            self.cp_ffc_sc = nn.Linear(in_dim, dim)
            self.bn_sc = nn.BatchNorm2d(dim)
            
            if self.stride == 2:
                self.avgpool = nn.AvgPool2d(2)
#                self.avg = nn.functional.interpolate()
#                self.maxpool = nn.MaxPool2d(2)
#                self.avgpool = nn.Sequential(
#                        nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=2, bias=False),
#                        nn.BatchNorm2d(dim),
#                        nn.ReLU(),
#                            )
                
                if up == True and stride == 2:
                    self.avgpool = nn.Upsample(scale_factor=2)
#                    self.maxpool = nn.MaxUnpool2d(2)
#                    self.avgpool = nn.Sequential(
#                            nn.ConvTranspose2d(dim, dim, kernel_size=3, padding=1, stride=2, bias=False),
#                            nn.BatchNorm2d(dim),
#                            nn.ReLU(),
#                            )
            
    def forward(self, x, sc_x):    
       
        _, c, w, h = x.shape
        out = rearrange(x, 'b c w h -> b w h c', c=c, w=w, h=h)
        out = self.cp_ffc(out)
        out = rearrange(out, 'b w h c-> b c w h', c=self.dim, w=w, h=h)
        out = self.bn(out)  
       
        if out.shape == sc_x.shape:
            if self.sc_x == True:
                out = sc_x + out
            if self.final == True:
                return out
#            out = F.layer_norm(out, out.size()[1:])
            
        else:
#            out = F.layer_norm(out, out.size()[1:])
            if self.sc_x == True:
                x = sc_x
            
        if self.stride == 2 or self.same == True:
            if self.sc_x == True:
                _, c, w, h = x.shape
                x = rearrange(x, 'b c w h -> b w h c', c=c, w=w, h=h)
                x = self.cp_ffc_sc(x)
                x = rearrange(x, 'b w h c-> b c w h', c=self.dim, w=w, h=h)
                x = self.bn_sc(x)
                out = out + x 
#                out = F.layer_norm(out, out.size()[1:])
            if self.same == True:
                return out
            
            _, _, h, w = out.shape
            if h % 2 == 0:
                out = self.avgpool(out)                    
            else:
                out = F.avg_pool2d(out, kernel_size=3, stride=2, padding=1)
                       
#            if self.up == False:               
#            out =   self.avgpool(out)# + self.maxpool(out)
#                out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=True)
                
        return out
    
class upa_lite_block(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1, cat=False, same=False, w=1, l=2, up=False, cpa=True, name=None):
        
        super(upa_lite_block, self).__init__()
        
        self.cat = cat
        self.stride = stride
        self.planes = planes
        self.same = same
        self.cpa = cpa
        self.name = name
        self.up = up
        
        self.att = CPA_lite(in_planes, planes, stride, same=same, up=up)
            
        if self.stride == 1:
            
            self.cnn = nn.Sequential(
                nn.Conv2d(in_planes, int(planes * w), kernel_size=3, padding=1, bias=False, groups=1),
                nn.BatchNorm2d(int(planes * w)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(planes * w), planes, kernel_size=3, padding=1, bias=False, groups=1),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)
                )#.cuda()
            if l == 1:
                w = 1
                self.cnn = nn.Sequential(
                    nn.Conv2d(in_planes, int(planes * w), kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(int(planes * w)),
                    nn.ReLU(),
                    )#.cuda()
                
#        if cpa == True:
#            self.att = CPA(in_planes, planes, stride, same=same, up=up)
                
        
            
    def forward(self, x):

        if self.stride == 2:
             out = self.att(x, x)
             
             return out
         
        else:
            out = self.cnn(x)
            out = self.att(x, out)

        if self.cat == True:
            out = torch.cat([x, out], 1)
            
        return out


class CPA_lite(nn.Module):
    '''Channel Pixel Attention'''
    
#      *same=False:
#       This scenario can be easily embedded after any CNNs, if size is same.
#        x (OG) ---------------
#        |                    |
#        sc_x (from CNNs)     CPA(x)
#        |                    |
#        out + <---------------
#        
#      *same=True:
#       This can be embedded after the CNNs where the size are different.
#        x (OG) ---------------
#        |                    |
#        sc_x (from CNNs)     |
#        |                    CPA(x)
#        CPA(sc_x)            |
#        |                    |
#        out + <---------------
#           
#      *sc_x=False
#       This operation can be seen a channel embedding with CPA
#       EX: x (3, 32, 32) => (16, 32, 32)
#        x (OG) 
#        |      
#        CPA(x)
#        |    
#        out 

    def __init__(self, in_dim, dim, stride=1, same=False, sc_x=True, up=False):
        
        super(CPA_lite, self).__init__()
            
        self.dim = dim
        self.stride = stride
        self.same = same
        self.sc_x = sc_x
        self.up = up
        
        self.cp_ffc = nn.Linear(in_dim, dim)#.cuda()
        self.bn = nn.BatchNorm2d(dim)#.cuda()

        if self.stride == 2 or self.same == True:
#            if sc_x == True:
#                self.cp_ffc_sc = nn.Linear(in_dim, dim)
#                self.bn_sc = nn.BatchNorm2d(dim)
            
            if self.stride == 2:
                self.avgpool = nn.AvgPool2d(2)
               
                
                if up == True and stride == 2:
                    self.avgpool = nn.Upsample(scale_factor=2)
                    
         
            
    def forward(self, x, sc_x):    
    
        _, c, w, h = x.shape
        out = rearrange(x, 'b c w h -> b w h c', c=c, w=w, h=h)
        out = self.cp_ffc(out)
        out = rearrange(out, 'b w h c-> b c w h', c=self.dim, w=w, h=h)
        out = self.bn(out)  
       
        if out.shape == sc_x.shape:
            if self.sc_x == True:
                out = sc_x + out
#            out = F.layer_norm(out, out.size()[1:])
#            
#        else:
#            out = F.layer_norm(out, out.size()[1:])
#            if self.sc_x == True:
#                x = sc_x
            
        if self.stride == 2 or self.same == True:
#            if self.sc_x == True:
#                _, c, w, h = x.shape
#                x = rearrange(x, 'b c w h -> b w h c', c=c, w=w, h=h)
#                x = self.cp_ffc_sc(x)
#                x = rearrange(x, 'b w h c-> b c w h', c=self.dim, w=w, h=h)
#                x = self.bn_sc(x)
#                out = out + x 
            
            if self.same == True:
                return out
            
            _, _, h, w = out.shape
            if self.up != True:
                if h % 2 == 0:
                    out = self.avgpool(out)                    
                else:
                    out = F.avg_pool2d(out, kernel_size=3, stride=2, padding=1)
           
            elif self.up == True:
                if h > 9:
                    out_up = self.avgpool(out)
                else:
                    out_up = F.upsample(out, scale_factor=1.9)
                    
                return out_up, out
            
        return out

   
class SPA(nn.Module):
    '''Spatial Pixel Attention'''

    def __init__(self, img, out=1):
        
        super(SPA, self).__init__()
        
        self.sp_ffc = nn.Sequential(
            nn.Linear(img**2, out**2)
            )   
        
    def forward(self, x):
        
        _, c, w, h = x.shape          
        x = rearrange(x, 'b c w h -> b c (w h)', c=c, w=w, h=h)
        x = self.sp_ffc(x)
        _, c, l = x.shape        
        out = rearrange(x, 'b c (w h) -> b c w h', c=c, w=int(l**0.5), h=int(l**0.5))

        return out

class upanets(nn.Module):
    def __init__(self, block, num_blocks, filter_nums, num_classes=100, img=32):
        
        super(upanets, self).__init__()
        
        num_classes = num_classes - 1
        self.in_planes = filter_nums
        self.filters = filter_nums
        w = 1
        
        self.root = nn.Sequential(
                nn.Conv2d(3, int(self.in_planes*w), kernel_size=3, padding=1, bias=False, groups=1),
                nn.BatchNorm2d(int(self.in_planes*w)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(self.in_planes*w), self.in_planes*1, kernel_size=3, padding=1, bias=False, groups=1),
                nn.BatchNorm2d(self.in_planes),
                nn.ReLU(inplace=True),
                )        
        self.emb = CPA(3, self.in_planes, same=True, stride=2)
        
#        self.layers = []
        
        self.layer1 = self._make_layer(block, int(self.filters*1), num_blocks[0], stride=2, name='layer1')
        self.layer2 = self._make_layer(block, int(self.filters*2), num_blocks[1], stride=2, name='layer2')
        self.layer3 = self._make_layer(block, int(self.filters*4), num_blocks[2], stride=2, name='layer3')
        self.layer4 = self._make_layer(block, int(self.filters*8), num_blocks[3], stride=2, name='layer4')

        self.root_up = nn.Sequential(
                nn.Conv2d(int(self.filters*31), int(self.filters*1), kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(int(self.filters*1)),
                nn.ReLU(),
                nn.Conv2d(int(self.filters*1), num_classes, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_classes),
                nn.ReLU(),
                )
        self.emb_up = CPA(int(self.filters*31), num_classes, stride=1, up=False, same=True)
        
#        self.layer5 = self._make_layer(block, int(self.filters*16), num_blocks[3], stride=2, name='layer5')
#        self.layer6 = self._make_layer(block, int(self.filters*32), num_blocks[3], stride=2, name='layer6')
        
#        self.fpn5 = self._make_layer(block, int(self.filters*2), num_blocks[3], stride=2, name='fpn5')
#        self.fpn6 = self._make_layer(block, int(self.filters*2), num_blocks[3], stride=2, name='fpn6')        
#        self.fpn0 = upa_block(int(self.filters*31), 256, stride=1)
#        self.fpn1 = upa_block(int(self.filters*31), 256, stride=1)
#        self.fpn2 = upa_block(int(self.filters*31), 256, stride=1)
#        self.fpn3 = upa_block(int(self.filters*31), 256, stride=1)
#        self.fpn4 = upa_block(int(self.filters*31), 256, stride=1)

        self.cls_layer1 = nn.Linear(1, 21)
        self.box_layer1 = nn.Linear(1, 4)
        self.ins_layer1 = nn.Linear(1, 20)

        self.cls_layer2 = nn.Linear(1, 21)
        self.box_layer2 = nn.Linear(1, 4)
        self.ins_layer2 = nn.Linear(1, 20)

        self.cls_layer3 = nn.Linear(1, 21)
        self.box_layer3 = nn.Linear(1, 4)
        self.ins_layer3 = nn.Linear(1, 20)
#        self.fpn5 = upa_block(256, 256, stride=2)
#        self.fpn6 = upa_block(256, 256, stride=2)
#        
#        self.layer6_up = self._make_layer(block, int(self.filters*4), num_blocks[3], 2, name='layer6_up')
#        self.layer5_up = self._make_layer(block, int(self.filters*4), num_blocks[2], 2, name='layer5_up')
#        self.layer4_up = self._make_layer(block, int(self.filters*4), num_blocks[1], 2, name='layer4_up')
#        self.layer3_up = self._make_layer(block, int(self.filters*4), num_blocks[0], 2, name='layer3_up')
#        self.layer2_up = self._make_layer(block, int(self.filters*4), num_blocks[0], 2, name='layer2_up')
#        self.layer1_up = self._make_layer(block, int(self.filters*4), num_blocks[0], 2, name='layer1_up')
#        self.out_up = self._make_layer(block, int(self.filters*1), num_blocks[0], 2, name='out_up')
        
#        self.layer6_up = upa_block(self.filters*64, 256, stride=2, up=True)
#        self.layer5_up = upa_block(self.filters*32 + 256, 256, stride=2, up=True)
#        self.layer4_up = upa_block(self.filters*16 + 256, 256, stride=2, up=True)
#        self.layer3_up = upa_block(self.filters*8 + 256, 256, stride=2, up=True)
#        self.layer2_up = upa_block(self.filters*4 + 256, 256, stride=2, up=True)
#        self.layer1_up = upa_block(self.filters*2 + 256, 256, stride=2, up=True)
#        self.fpn_68 = upa_block(self.filters*4, 3*4, l=1)
#        self.fpn_34 = upa_block(self.filters*8, 3*4, l=1)
#        self.fpn_17 = upa_block(self.filters*16, 3*4, l=1)
#        self.fpn_9 = upa_block(self.filters*32, 3*4, l=1)
#        self.fpn_5 = upa_block(self.filters*64, 3*4, l=1)
        
#        self.root_up = nn.Sequential(
#                nn.Conv2d(int(self.filters*31), int(self.filters*1), kernel_size=3, padding=1, bias=False),
#                nn.BatchNorm2d(int(self.filters*1)),
#                nn.ReLU(),
#                nn.Conv2d(int(self.filters*1), 1, kernel_size=3, padding=1, bias=False),
#                nn.BatchNorm2d(1),
#                nn.ReLU(),
#                )
#        self.emb_up = CPA(int(self.filters*31), 1, same=True)
        
#        self.spa0 = SPA(img)
#        self.spa1 = SPA(img)
#        self.spa2 = SPA(int(img*0.5))
#        self.spa3 = SPA(int(img*0.25))
#        self.spa4 = SPA(int(img*0.125))

#        self.linear = nn.Linear(int(self.filters*31), num_classes)
#        self.bn = nn.BatchNorm1d(int(self.filters*31))
        
#        self.linear0 = nn.Linear(int(self.filters*1), num_classes)
#        self.bn0 = nn.BatchNorm1d(int(self.filters*1))
#        self.linear1 = nn.Linear(int(self.filters*2), num_classes)
#        self.bn1 = nn.BatchNorm1d(int(self.filters*2))
#        self.linear2 = nn.Linear(int(self.filters*4), num_classes)
#        self.bn2 = nn.BatchNorm1d(int(self.filters*4))
#        self.linear3 = nn.Linear(int(self.filters*8), num_classes)
#        self.bn3 = nn.BatchNorm1d(int(self.filters*8))
#        self.linear4 = nn.Linear(int(self.filters*16), num_classes)
#        self.bn4 = nn.BatchNorm1d(int(self.filters*16))
     
#        self.cpa_pred = nn.Linear(5, 1, bias=False)
#    def _make_layer(self, block, planes, num_blocks, stride):
#        '''w/o upsampling'''
#        strides = [stride] + [1]*(num_blocks - 1)
#        layers = []
#        self.planes = planes
#        planes = planes // num_blocks
#
#        for i, stride in enumerate(strides):
#            
#            if i == 0 and stride == 1:
#                layers.append(block(self.planes, self.planes, stride, same=True))
#                strides.append(1)
#                self.in_planes = self.planes
#                
#            elif i != 0 and stride == 1:
#                layers.append(block(self.in_planes, planes, stride, cat=True))                
#                self.in_planes = self.in_planes + planes 
#                    
#            else:   
#                layers.append(block(self.in_planes, self.planes, stride))
#                strides.append(1)
#                self.in_planes = self.planes
#                
#        return nn.Sequential(*layers)
    
    def _make_layer(self, block, planes, num_blocks, stride, name=None):
        '''w/ upsampling'''
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        self.planes = planes
        planes = planes // num_blocks
#        print('='*10)
#        print(name)
        for i, stride in enumerate(strides):

            if 'up' in name:
                if i != 0 and stride == 1:
                   
                    layers.append(block(self.in_planes, planes, stride, cat=True, up=True))                
                    self.in_planes = self.in_planes + planes 
                        
                else:   
                    if name != 'layer5_up':
                        self.in_planes = self.in_planes * 2 
                        
                    layers.append(block(self.in_planes, self.planes, stride, up=True))
                    strides.append(1)
                    self.in_planes = self.planes
            
            else:
                if i == 0 and stride == 1:
                    layers.append(block(self.planes, self.planes, stride, same=True, name=name))
                    strides.append(1)
                    self.in_planes = self.planes
                    
                elif i != 0 and stride == 1:
                    layers.append(block(self.in_planes, planes, stride, cat=True, name=name))                
                    self.in_planes = self.in_planes + planes 
                        
                else:   
                    layers.append(block(self.in_planes, self.planes, stride, name=name))
                    strides.append(1)
                    self.in_planes = self.planes
#            print('out:', self.in_planes)
        return nn.Sequential(*layers)

    def forward(self, x):
        
#        with timer.env('backbone'):
#        b = x.shape[0]
#        img_size = x.shape[2:]
        out01 = self.root(x)
        out0 = self.emb(x, out01)
        out0 = F.avg_pool2d(out0, kernel_size=2, stride=2)
#        img_size = out0.shape[2:]
        img_size=(68,68)
        
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        
#        out5 = self.layer5(out4)
#        out6 = self.layer6(out5)
##
#        print('out0: ', out0.shape)
#        print('out1: ', out1.shape)
#        print('out2: ', out2.shape)
#        print('out3: ', out3.shape)
#        print('out4: ', out4.shape)
##        print('out5: ', out5.shape)
##        print('out6: ', out6.shape)
#        
#        out6_up, out6 = self.layer6_up(out6)
##        out6_up, out5 = size_check(out6, out5)
##        out6_up = out6
#        out5_up, out5 = self.layer5_up(torch.cat([out6_up, out5],1))
##        out5_up, out4 = size_check(out5, out4)
##        out5_up = out5
#        out4_up, out4 = self.layer4_up(torch.cat([out5_up, out4],1))
##        out4_up, out3 = size_check(out4, out3)
##        out4_up = out4
#        out3_up, out3 = self.layer3_up(torch.cat([out4_up, out3],1))
##        out3_up, out2 = size_check(out3, out2)
##        out3_up = out3
#        out2_up, out2 = self.layer2_up(torch.cat([out3_up, out2],1))
##        out2_up, out1 = size_check(out2, out1)
##        out2_up = out2
#        out1_up, out1 = self.layer1_up(torch.cat([out2_up, out1],1))
        
#        out0 = np.copy(out0.cpu(), order='C')
#        out1 = np.copy(out1.cpu(), order='C')
#        out2 = np.copy(out2.cpu(), order='C')
#        out3 = np.copy(out3.cpu(), order='C')
#        out4 = np.copy(out4.cpu(), order='C')
        
        cls_preds = []
        box_preds = []
        ins_preds = []
        for outi in [out0, out1,out2, out3, out4]:
            
            outi = F.avg_pool2d(outi,kernel_size=3,padding=1,stride=2)
            outi = F.avg_pool2d(outi,kernel_size=3,padding=1,stride=2)
            
            _, c, w, h = outi.shape
            outi = rearrange(outi, 'b c w h -> b (w h) c', c=c, w=w, h=h)
            outi = outi.mean(-1, True)
            
            cls_pred1 = self.cls_layer1(outi)
            box_pred1 = self.box_layer1(outi)
            ins_pred1 = self.ins_layer1(outi)

            cls_pred2 = self.cls_layer1(outi)
            box_pred2 = self.box_layer1(outi)
            ins_pred2 = self.ins_layer1(outi)

            cls_pred3 = self.cls_layer1(outi)
            box_pred3 = self.box_layer1(outi)
            ins_pred3 = self.ins_layer1(outi)
            
            cls_preds.append(cls_pred1)
            cls_preds.append(cls_pred2)
            cls_preds.append(cls_pred3)
            
            box_preds.append(box_pred1)
            box_preds.append(box_pred2)
            box_preds.append(box_pred3)
            
            ins_preds.append(ins_pred1)
            ins_preds.append(ins_pred2)
            ins_preds.append(ins_pred3)
            
        cls_preds = torch.cat(cls_preds, 1)
        box_preds = torch.cat(box_preds, 1)
        ins_preds = torch.cat(ins_preds, 1)

        out0 = F.interpolate(out0, size=img_size, mode='bilinear', align_corners=False)
        out1 = F.interpolate(out1, size=img_size, mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, size=img_size, mode='bilinear', align_corners=False)
        out3 = F.interpolate(out3, size=img_size, mode='bilinear', align_corners=False)
        out4 = F.interpolate(out4, size=img_size, mode='bilinear', align_corners=False)

        out0_ = torch.cat([
                out0, 
                out1, out2, out3, out4,# x,
#                out1_up,
#                out2_up, out3_up, out4_up
                ], 1)
        
#        print('out combine: ',out_.shape)
        out0_up = self.root_up(out0_)
        out = self.emb_up(out0_, out0_up)    
#        print('out: ', out.shape)
#        '''extended parts'''
        
#        out0 = F.interpolate(out0, size=img_size, mode='bilinear', align_corners=False)
#        out1_ = F.interpolate(out0_, size=(34,34), mode='bilinear', align_corners=False)
#        out2_ = F.interpolate(out0_, size=(17,17), mode='bilinear', align_corners=False)
#        out3_ = F.interpolate(out0_, size=(9,9), mode='bilinear', align_corners=False)
#        out4_ = F.interpolate(out0_, size=(5,5), mode='bilinear', align_corners=False)

#        out0 = self.fpn0(out0_)
#        out1 = self.fpn1(out1_)
#        out2 = self.fpn2(out2_)
#        out3 = self.fpn3(out3_)
#        out4 = self.fpn4(out4_)
        
#        out0 = out0_
#        out1 = out1_
#        out2 = out2_
#        out3 = out3_
#        out4 = out4_
#        out5 = self.fpn5(out4)
#        out6 = self.fpn6(out5)
    
#            out = torch.cat([
#                    out2.reshape(b, 256, -1),
#                    out3.reshape(b, 256, -1), 
#                    out4.reshape(b, 256, -1),
#                    out5.reshape(b, 256, -1),
#                    out6.reshape(b, 256, -1)
#                    ], 2)
            
#        print('out0: ', out0.shape)
#        print('out1: ', out1.shape)
#        print('out2: ', out2.shape)
#        print('out3: ', out3.shape)
#        print('out4: ', out4.shape)
#        print('out5: ', out5.shape)
#        print('out6: ', out6.shape)
#            print('out: ', out.shape)
            
#            out_68 = self.tmp_68(out2)
#            out_34 = self.tmp_34(out3)
#            out_17 = self.tmp_17(out4)
#            out_9 = self.tmp_9(out5)
#            out_5 = self.tmp_5(out6)
#         
#            print('out68: ', out_68.shape)
#            print('out34: ', out_34.shape)
#            print('out17: ', out_17.shape)
#            print('out9: ', out_9.shape)
#            print('out5: ', out_5.shape)
            
#            out4_up = self.layer4_up(out4)
#            print('up out4:', out4_up.shape)
#            out4_up, out3 = size_check(out4_up, out3)
#            out3_up = self.layer3_up(torch.cat([out4_up, out3], 1))
#            print('up out3:', out3_up.shape)
#            out3_up, out2 = size_check(out3_up, out2)
#            out2_up = self.layer2_up(torch.cat([out3_up, out2], 1))
#            out2_up, out1 = size_check(out2_up, out1)
#            out1_up = self.layer1_up(torch.cat([out2_up, out1], 1))

            
            
#            print('up out2:', out2_up.shape)
#            print('up out1:', out1_up.shape)
            
#            out0 = self.root_up(out)
#            out_sod = self.emb_up(out, out0)
        
#        out0_spa = self.spa0(out0)
#        out1_spa = self.spa1(out1)
#        out2_spa = self.spa2(out2)
#        out3_spa = self.spa3(out3)
#        out4_spa = self.spa4(out4)
        
#        out0_gap = F.avg_pool2d(out0, out0.size()[2:])
#        out1_gap = F.avg_pool2d(out1, out1.size()[2:])
#        out2_gap = F.avg_pool2d(out2, out2.size()[2:])
#        out3_gap = F.avg_pool2d(out3, out3.size()[2:])
#        out4_gap = F.avg_pool2d(out4, out4.size()[2:])
###      
#        out0 = out0_gap# + out0_spa
#        out1 = out1_gap# + out1_spa
#        out2 = out2_gap# + out2_spa
#        out3 = out3_gap# + out3_spa
#        out4 = out4_gap# + out4_spa
#        
#        out0 = F.layer_norm(out0, out0.size()[1:])
#        out1 = F.layer_norm(out1, out1.size()[1:])
#        out2 = F.layer_norm(out2, out2.size()[1:])
#        out3 = F.layer_norm(out3, out3.size()[1:])
#        out4 = F.layer_norm(out4, out4.size()[1:])
        
#        out0 = out0.view(out0.size(0), -1)
#        out1 = out1.view(out1.size(0), -1)
#        out2 = out2.view(out2.size(0), -1)
#        out3 = out3.view(out3.size(0), -1)
#        out4 = out4.view(out4.size(0), -1)
        
#        out0 = self.bn0(out0)
#        out1 = self.bn1(out1)
#        out2 = self.bn2(out2)
#        out3 = self.bn3(out3)
#        out4 = self.bn4(out4)
        
#        out0 = self.linear0(out0)
#        out1 = self.linear1(out1)
#        out2 = self.linear2(out2)
#        out3 = self.linear3(out3)
#        out4 = self.linear4(out4)
#        
#        out = torch.stack([out4, out3, out2, out1, out0], 1)
        
#        b, c, l = out.shape
#        out = rearrange(out, 'b c l -> b l c', c=c, l=l)
#        out = self.cpa_pred(out)
#        out = rearrange(out, 'b l c-> b c l', c=1, l=l)
#        out = out.squeeze()
#        
#        out = torch.cat([out4, out3, out2, out1, out0], 1)
#        out = out.view(out.size(0), -1)
#        out = self.bn(out) # please exclude when using the test function
#        out = self.linear(out)

#        return (out1, out2, out3 ,out4)
#        return (out1, out2, out3 ,out4, out5, out6)
        return out, cls_preds, box_preds, ins_preds

def size_check(x1, x2, label=False):

    if x1.shape[-2:] != x2.shape[-2:]:
        if int(x1.shape[-2] * x1.shape[-1]) > int(x2.shape[-2] * x2.shape[-1]):
            x2 = F.interpolate(x2, size=x1.shape[-2:], mode='bilinear', align_corners=False)
            
        else:
            x1 = F.interpolate(x1, size=x2.shape[-2:], mode='bilinear', align_corners=False)
            
    return x1, x2

def UPANets(f, c = 100, block = 1, img = 32):
    
    return upanets(upa_lite_block, [int(4*block), int(4*block), int(4*block), int(4*block)], f, num_classes=c, img=img)

def test():
    
#    net = UPANets(16, 10, 1, 32)
#    y = net(torch.randn(1, 3, 32, 32))
    
    net = UPANets(64, 21)
    y = net(torch.randn(1, 3, 544, 544))
    
    return y

##%%
#import time
#
##net =  UPANets(32, 10, 1, 32)
#net = net.cuda()
#x = torch.randn(1, 3, 544, 544).cuda()
#
##%%
#start = time.time()
#y = net(x)
#end = time.time() - start
#fps = 1/ end
##%%
#from torchsummary import summary
#net = UPANets(64, 21, 1, 544)
#summary(net, (3, 544, 544))

#net = Yolact(cfg)
#net = net.cuda()
#x = torch.zeros((1, 3, 544, 544))#.cuda()
#y = net(x)
#avg = MovingAverage()
#timer.reset()
#with timer.env('everything else'):
#    net(x)
#avg.add(timer.total_time())
#print('\033[2J') # Moves console cursor to 0,0
#timer.print_stats()
#print('Avg fps: %.2f\tAvg ms: %.2f         ' % (1/avg.get_avg(), avg.get_avg()*1000))