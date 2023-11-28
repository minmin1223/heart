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
    
    def __init__(self, in_planes, planes, stride=1, cat=False, same=False, w=2, l=2, up=False, k=1, bn=True):
        
        super(upa_block, self).__init__()
        
        self.cat = cat
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        self.same = same
        self.cnn = nn.Sequential(
            nn.Conv2d(in_planes, int(planes * w), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(planes * w)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(planes * w), planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
            )
        if l == 1:
            w = 1
            if k==1 :
                
                self.cnn = nn.Sequential(
                    nn.Conv2d(in_planes, int(planes * w), kernel_size=k, bias=False),
                    nn.BatchNorm2d(int(planes * w)),
                    nn.ReLU(inplace=True),
                    )
            if k==3:
                self.cnn = nn.Sequential(
                    nn.Conv2d(in_planes, int(planes * w), kernel_size=k, padding=1, bias=False),
                    nn.BatchNorm2d(int(planes * w)),
                    nn.ReLU(inplace=True),
                    )
                
        self.att = CPA(in_planes, planes, stride, same=same, up=up, bn=True)
        
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
    
    def __init__(self, in_dim, dim, stride=1, same=False, up=False, sc_x=True, final=False, bn=True):
        
        super(CPA, self).__init__()
            
        self.dim = dim
        self.stride = stride
        self.same = same
        self.sc_x = sc_x
        self.final = final
        self.bn = bn
        
        self.cp_ffc = nn.Linear(in_dim, dim)
        if self.bn == True:
            self.bn = nn.BatchNorm2d(dim)
        self.up = up

        if self.stride == 2 or self.same == True:
            self.cp_ffc_sc = nn.Linear(in_dim, dim)
            self.bn_sc = nn.BatchNorm2d(dim)
            
            if self.stride == 2:
                self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
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
        if self.bn == True:
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
                if self.bn == True:
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
    
    def __init__(self, in_planes, planes, stride=1, cat=False, same=False, w=2, l=2, up=False, cpa=True, name=None):
        
        super(upa_lite_block, self).__init__()
        
        self.cat = cat
        self.stride = stride
        self.planes = planes
        self.same = same
        self.cpa = cpa
        self.name = name
        self.up = up
        
        if self.stride == 2:
            
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
                    nn.ReLU(inplace=True),
                    )#.cuda()
                
#        if cpa == True:
#            self.att = CPA(in_planes, planes, stride, same=same, up=up)
                
        
            
    def forward(self, x):

        if self.stride == 2:
             out = self.att(x, x)
             
             return out
         
        else:
            out = self.cnn(x)
#            out = self.att(x, out)

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
                self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
               
                
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
#                if h % 2 == 0:
                out = self.avgpool(out)                    
#                else:
#                    out = F.avg_pool2d(out, kernel_size=3, stride=2, padding=1)
           
            elif self.up == True:
                if h > 9:
                    out_up = self.avgpool(out)
                else:
                    out_up = F.upsample(out, scale_factor=1.9)
                    
                return out_up, out
            
        return out

   
class SPA(nn.Module):
    '''Spatial Pixel Attention'''

    def __init__(self, inp, out=1):
        
        super(SPA, self).__init__()
        
        self.sp_ffc = nn.Sequential(
            nn.Linear(inp, out)
            )   
        
    def forward(self, x):
        
        r=3
        _, o, c = x.shape   
        l = int(o/r)
        x = rearrange(x, 'b (r l) c -> b c r l', c=c, l=l, r=r)
        x = self.sp_ffc(x)
        _, c, r, l = x.shape        
        out = rearrange(x, 'b c r l -> b (r l) c', c=c, l=l, r=r)

        return out

#class SPA(nn.Module):
#    '''Spatial Pixel Attention'''
#
#    def __init__(self, img, out=1):
#        
#        super(SPA, self).__init__()
#        
#        self.sp_ffc = nn.Sequential(
#            nn.Linear(img**2, out**2)
#            )   
#        
#    def forward(self, x):
#        
#        _, c, w, h = x.shape          
#        x = rearrange(x, 'b c w h -> b c (w h)', c=c, w=w, h=h)
#        x = self.sp_ffc(x)
#        _, c, l = x.shape        
#        out = rearrange(x, 'b c (w h) -> b c w h', c=c, w=int(l**0.5), h=int(l**0.5))
#
#        return out

class upanets(nn.Module):
    def __init__(self, block, num_blocks, filter_nums, num_classes=100, img=32):
        
        super(upanets, self).__init__()
        
        num_classes = num_classes - 1
        self.in_planes = filter_nums
        self.filters = filter_nums
        w = 2
        ar = 3
        
#        self.root = nn.Sequential(
#                nn.Conv2d(3, int(self.in_planes*w), kernel_size=3, padding=1, bias=False, groups=1),
#                nn.BatchNorm2d(int(self.in_planes*w)),
#                nn.ReLU(inplace=True),
#                nn.Conv2d(int(self.in_planes*w), self.in_planes*1, kernel_size=3, padding=1, bias=False, groups=1),
#                nn.BatchNorm2d(self.in_planes),
#                nn.ReLU(inplace=True),
#                )        
#        self.emb = CPA(3, self.in_planes, same=False, stride=2)
        
        self.root = upa_block(3, self.in_planes, stride=2, same=False)
        
#        self.layers = []
        
        self.layer1 = self._make_layer(block, int(self.filters*1), num_blocks[0], stride=2, name='layer1')
        self.layer2 = self._make_layer(block, int(self.filters*2), num_blocks[1], stride=2, name='layer2')
        self.layer3 = self._make_layer(block, int(self.filters*4), num_blocks[2], stride=2, name='layer3')
        self.layer4 = self._make_layer(block, int(self.filters*8), num_blocks[3], stride=2, name='layer4')

#        self.fpns = []

        self.fpn0 = nn.Sequential(
                upa_block(int(self.filters*30), 256, stride=1, l=1, k=1, same=True, bn=False),
                upa_block(256, 256, stride=1, l=1, k=3, same=True, bn=False)
                )
        self.fpn1 = nn.Sequential(
                upa_block(int(self.filters*30), 256, stride=1, l=1, k=1, same=True, bn=False),
                upa_block(256, 256, stride=1, l=1, k=3, same=True, bn=False)
                )
        self.fpn2 = nn.Sequential(
                upa_block(int(self.filters*30), 256, stride=1, l=1, k=1, same=True, bn=False),
                upa_block(256, 256, stride=1, l=1, k=3, same=True, bn=False)
                )
        self.fpn3 = nn.Sequential(
                upa_block(int(self.filters*30), 256, stride=1, l=1, k=1, same=True, bn=False),
                upa_block(256, 256, stride=1, l=1, k=3, same=True, bn=False)
                )
        self.fpn4 = nn.Sequential(
                upa_block(int(self.filters*30), 256, stride=1, l=1, k=1, same=True, bn=False),
                upa_block(256, 256, stride=1, l=1, k=3, same=True, bn=False)
                )       
        self.fpn5 = nn.Sequential(
                upa_block(int(self.filters*30), 256, stride=1, l=1, k=1, same=True, bn=False),
                upa_block(256, 256, stride=1, l=1, k=3, same=True, bn=False)
                )

#        self.fpns.append(self.fpn0)
#        self.fpns.append(self.fpn1)
#        self.fpns.append(self.fpn2)
#        self.fpns.append(self.fpn3)
#        self.fpns.append(self.fpn4)
        
#        self.sem_ps = nn.Sequential(
#                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
#                nn.PixelShuffle(2),
#                nn.BatchNorm2d(64),
#                nn.ReLU(inplace=True),
#                )
#        self.sem_con = upa_block(256, 256, stride=1, l=1, k=3, same=True)
        self.sem_head = upa_block(256, num_classes, stride=1, same=True)
        
#        self.sem_heads = []
#        self.sem_heads.append(self.sem_68_head = upa_block(256, num_classes, stride=1, l=1, k=3, same=True))
#        self.sem_heads.append(self.sem_34_head = upa_block(256, num_classes, stride=1, l=1, k=3, same=True))
#        self.sem_heads.append(self.sem_17_head = upa_block(256, num_classes, stride=1, l=1, k=3, same=True))
#        self.sem_heads.append(self.sem_09_head = upa_block(256, num_classes, stride=1, l=1, k=3, same=True))
#        self.sem_heads.append(self.sem_05_head = upa_block(256, num_classes, stride=1, l=1, k=3, same=True))

#        self.ins_ps = nn.Sequential(
#                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
#                nn.PixelShuffle(2),
#                nn.BatchNorm2d(64),
#                nn.ReLU(inplace=True),
#                )
#        self.ins_con = upa_block(256, 256, stride=1, l=1, k=3, same=True)
        self.ins_head = upa_block(256, num_classes, stride=1, same=True)
        
#        self.ins_heads = []
#        self.ins_heads.append(self.ins_68_head = upa_block(256, num_classes, stride=1, l=1, k=3, same=True))
#        self.ins_heads.append(self.ins_34_head = upa_block(256, num_classes, stride=1, l=1, k=3, same=True))
#        self.ins_heads.append(self.ins_17_head = upa_block(256, num_classes, stride=1, l=1, k=3, same=True))
#        self.ins_heads.append(self.ins_34_head = upa_block(256, num_classes, stride=1, l=1, k=3, same=True))
#        self.ins_heads.append(self.ins_05_head = upa_block(256, num_classes, stride=1, l=1, k=3, same=True))
        
#        self.cls_head_con = nn.Sequential(
#                nn.Conv2d(256, int((num_classes+1)*ar), kernel_size=3, padding=1, bias=False),
#                nn.BatchNorm2d(int((num_classes+1)*ar)),
#                nn.ReLU(inplace=True),
#                )
#        self.cls_head_cpa = CPA(256, int((num_classes+1)*ar), stride=1, up=False, same=True)
        self.cls_head = upa_block(256 , int((num_classes+1)*ar), stride=1, same=True)
        
#        self.cls_68_head = upa_block(256 , int((num_classes+1)*ar), stride=1, l=1, k=3, same=True)
#        self.cls_34_head = upa_block(256 , int((num_classes+1)*ar), stride=1, l=1, k=3, same=True)
#        self.cls_17_head = upa_block(256 , int((num_classes+1)*ar), stride=1, l=1, k=3, same=True)
#        self.cls_09_head = upa_block(256 , int((num_classes+1)*ar), stride=1, l=1, k=3, same=True)
#        self.cls_05_head = upa_block(256 , int((num_classes+1)*ar), stride=1, l=1, k=3, same=True)
#        self.cls_heads = nn.ModuleList([upa_block(256, int((num_classes+1)*ar), stride=1, l=1, k=3, same=True) for i in range(5)])

#        self.box_head_con = nn.Sequential(
#                nn.Conv2d(256, int(4*ar), kernel_size=3, padding=1, bias=False),
#                nn.BatchNorm2d(int(4*ar)),
#                nn.ReLU(inplace=True),
#                )
#        self.box_head_cpa = CPA(256, int(4*ar), stride=1, up=False, same=True)
        self.box_head = upa_block(256 , int(4*ar), stride=1, same=True)
        
#        self.box_68_head = upa_block(256 , int(4*ar), stride=1, l=1, k=3, same=True)
#        self.box_34_head = upa_block(256 , int(4*ar), stride=1, l=1, k=3, same=True)
#        self.box_17_head = upa_block(256 , int(4*ar), stride=1, l=1, k=3, same=True)
#        self.box_09_head = upa_block(256 , int(4*ar), stride=1, l=1, k=3, same=True)
#        self.box_05_head = upa_block(256 , int(4*ar), stride=1, l=1, k=3, same=True)
#        self.box_heads = nn.ModuleList([upa_block(256, int(4*ar), stride=1, l=1, k=3, same=True) for i in range(5)])
#        self.coe_head_con = nn.Sequential(
#                nn.Conv2d(256, int(ar*num_classes), kernel_size=3, padding=1, bias=False),
#                nn.BatchNorm2d(int(ar*num_classes)),
#                nn.ReLU(inplace=True),
#                )
#        self.coe_head_cpa = CPA(256, int(ar*num_classes), stride=1, up=False, same=True)
        self.coe_head = upa_block(256 , int(ar*num_classes), stride=1, same=True)
        
#        self.coe_68_head = upa_block(256 , int(ar*num_classes), stride=1, l=1, k=3, same=True)
#        self.coe_34_head = upa_block(256 , int(ar*num_classes), stride=1, l=1, k=3, same=True)
#        self.coe_17_head = upa_block(256 , int(ar*num_classes), stride=1, l=1, k=3, same=True)
#        self.coe_09_head = upa_block(256 , int(ar*num_classes), stride=1, l=1, k=3, same=True)
#        self.coe_05_head = upa_block(256 , int(ar*num_classes), stride=1, l=1, k=3, same=True)
#        self.coe_heads = nn.ModuleList([upa_block(256, int(ar*num_classes), stride=1, l=1, k=3, same=True) for i in range(5)])
#        self.cls_layer = nn.Linear((num_classes+1), (num_classes+1))
#        self.box_layer = nn.Linear(4, 4)
#        self.coe_layer = nn.Linear(num_classes, num_classes)
        
        self.tanh = torch.nn.Tanh()
    
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
        out0 = self.root(x)
#        out0 = self.emb(x, out01)
#        out0 = F.avg_pool2d(out0, kernel_size=3, stride=2, padding=1)
#        img_size = out0.shape[2:]
#        img_size=(68,68)
        
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        
#        for i in [out0, out1, out2, out3, out4]:
#            print(i.shape)
#        
#        return out1, out2, out3 ,out4
        
        outs = [out1, out2, out3, out4]
        out_136s = []
        out_68s = []
        out_34s = []
        out_17s = []
        out_09s = []
        out_05s = []
        
        
        for i in range(len(outs)):
            out_136s.append(F.interpolate(outs[i], size=(136 ,136), mode='bilinear', align_corners=False))
            out_68s.append(F.interpolate(outs[i], size=(68 ,68), mode='bilinear', align_corners=False))
            out_34s.append(F.interpolate(outs[i], size=(34, 34), mode='bilinear', align_corners=False))
            out_17s.append(F.interpolate(outs[i], size=(17, 17), mode='bilinear', align_corners=False))
            out_09s.append(F.interpolate(outs[i], size=(9, 9), mode='bilinear', align_corners=False))
            out_05s.append(F.interpolate(outs[i], size=(5, 5), mode='bilinear', align_corners=False))
        
        out_136s = torch.cat(out_136s, 1)
        out_68s = torch.cat(out_68s, 1)
        out_34s = torch.cat(out_34s, 1)
        out_17s = torch.cat(out_17s, 1)
        out_09s = torch.cat(out_09s, 1)
        out_05s = torch.cat(out_05s, 1)
            
        out_68 = self.fpn0(out_68s)
        out_34 = self.fpn1(out_34s)
        out_17 = self.fpn2(out_17s)
        out_09 = self.fpn3(out_09s)
        out_05 = self.fpn4(out_05s)

        
#        return (out_68, out_34, out_17, out_09, out_05)
        
#        out_136 = F.interpolate(out_68, size=(136, 136), mode='bilinear', align_corners=False)
        
        out_136 = self.fpn5(out_136s)
        
#        sem_ps = self.sem_ps(out_68)
#        sem_out = self.sem_con(sem_ps)
        sem_out = self.sem_head(out_68)
        
#        ins_ps = self.ins_ps(out_68)
#        ins_out = self.ins_con(out_136)
        ins_out = self.ins_head(out_136).permute(0, 2, 3, 1).contiguous()
        
#        ins_out = sem_out.permute(0, 2, 3, 1).contiguous()
        
        cls_out, box_out, coe_out = [], [], []
        for i, outi in enumerate([out_68, out_34, out_17, out_09, out_05]):
#            outi = self.fpn(outi)
            
#            print('outi: ', outi.shape)
#            cls_outi = self.cls_head_con(outi)
#            cls_outi = self.cls_head_cpa(outi, cls_outi).permute(0, 2, 3, 1).reshape(outi.size(0), -1, 21)
            cls_outi = self.cls_head(outi).permute(0, 2, 3, 1).reshape(outi.size(0), -1, 21)
            cls_out.append(cls_outi)
                       
#            box_outi = self.box_head_con(outi)
#            box_outi = self.box_head_cpa(outi, box_outi).permute(0, 2, 3, 1).reshape(outi.size(0), -1, 4)
            box_outi = self.box_head(outi).permute(0, 2, 3, 1).reshape(outi.size(0), -1, 4)
            box_out.append(box_outi)
           
#            coe_outi = self.coe_head_con(outi)
#            coe_outi = self.coe_head_cpa(outi, coe_outi).permute(0, 2, 3, 1).reshape(outi.size(0), -1, 20)
            coe_outi = self.coe_head(outi).permute(0, 2, 3, 1).reshape(outi.size(0), -1, 20)
            coe_out.append(coe_outi)
            
#            if i == 0:
#                sem_out = self.sem_root_up(outi)
#                sem_out = self.sem_emb_up(outi, sem_out)
#                
#                ins_out = self.ins_root_up(outi)
#                ins_out = self.ins_emb_up(outi, ins_out).permute(0, 2, 3, 1).contiguous()
        
        cls_out = torch.cat(cls_out, 1)
        box_out = torch.cat(box_out, 1)
        coe_out = torch.cat(coe_out, 1)

#        cls_out = self.cls_layer(cls_out)
#        box_out = self.box_layer(box_out)
#        coe_out = self.coe_layer(coe_out)
        
        coe_out = torch.tanh(coe_out)

        return sem_out, ins_out, cls_out, box_out, coe_out

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
    sem_out, ins_out, cls_out, box_out, coe_out = net(torch.randn(1, 3, 544, 544))
    
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