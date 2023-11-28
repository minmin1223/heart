#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

    
class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret
    
class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, coord=True, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        
        self.coord = coord
        
        in_size = in_channels+2
        
        if self.coord == False:
            in_size = in_channels
            
        if with_r:
            in_size += 1
        # self.conv = nn.Conv2d(in_size, out_channels, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv = nn.Conv2d(in_size, out_channels, kernel_size=1, bias=False, dilation=2)
        # self.conv = upa_block(in_size, out_channels, stride=1, l=1, k=1, same=False, bn=False, cpa=True, act=False)

    def forward(self, x):
        
        if self.coord == True:
            ret = self.addcoords(x)
        else:
            ret = x
        
#        print('ret: ', ret.shape)
        ret = self.conv(ret)
        return ret
    
class upa_block(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1, cat=False, same=False, w=2, l=2, up=False, k=1, bn=True, cpa=True, act=True):
        
        super(upa_block, self).__init__()
        
        self.cat = cat
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        self.same = same
        self.bn = bn
        self.cpa = cpa
        self.act = act
        
        if self.bn == True:
            
            self.cnn = nn.Sequential(
                nn.Conv2d(in_planes, int(planes * w), kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(int(planes * w)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(planes * w), planes, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)
                )
            
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(in_planes, int(planes * w), kernel_size=3, padding=1, bias=False),
#                nn.BatchNorm2d(int(planes * w)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(planes * w), planes, kernel_size=3, padding=1, bias=False),
#                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)
                )
            
        if l == 1:
            w = 1
            if k==1 :
                
                self.cnn = nn.Sequential(
                    nn.Conv2d(in_planes, int(planes * w), kernel_size=k, bias=False),
#                    nn.BatchNorm2d(int(planes * w)),
                    nn.ReLU(inplace=True),
                    )
                
                if self.act == False:
                    self.cnn = nn.Sequential(
                    nn.Conv2d(in_planes, int(planes * w), kernel_size=k, bias=False),
#                    nn.BatchNorm2d(int(planes * w)),
#                    nn.ReLU(inplace=True),
                    )
                    
            if k==3:
                self.cnn = nn.Sequential(
                    nn.Conv2d(in_planes, int(planes * w), kernel_size=k, padding=1, bias=False),
#                    nn.BatchNorm2d(int(planes * w)),
                    nn.ReLU(inplace=True),
                    )
                
                if self.act == False:
                    self.cnn = nn.Sequential(
                    nn.Conv2d(in_planes, int(planes * w), kernel_size=k, padding=1, bias=False),
#                    nn.BatchNorm2d(int(planes * w)),
#                    nn.ReLU(inplace=True),
                    )
        
        if self.cpa == True:
            self.att = CPA(in_planes, planes, stride, same=same, up=up, bn=bn)
        
    def forward(self, x):
#        print('in_planes:', self.in_planes)
#        print('planes:', self.planes)
        out = self.cnn(x)
        
        if self.cpa == True:
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
        
        self.cp_ffc = nn.Linear(in_dim, dim, bias=False)
        if self.bn == True:
            self.bn = nn.BatchNorm2d(dim)
        self.up = up

        if self.stride == 2 or self.same == True:
            self.cp_ffc_sc = nn.Linear(in_dim, dim, bias=False)
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
        
        self.cp_ffc = nn.Linear(in_dim, dim, bias=False)#.cuda()
        self.bn = nn.BatchNorm2d(dim)#.cuda()

        if self.stride == 2 or self.same == True:
            
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
            
        if self.stride == 2 or self.same == True:
            
            if self.same == True:
                return out
            
            _, _, h, w = out.shape
            if self.up != True:
                out = self.avgpool(out)                    
           
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

class upanets(nn.Module):
    def __init__(self, block, num_blocks, filter_nums, num_classes=100, img=32):
        
        super(upanets, self).__init__()
        
        # num_classes = num_classes - 1
        num_classes = 4
        self.in_planes = filter_nums
        self.filters = filter_nums
        ar = 1
        
        self.root = upa_block(int(3), self.in_planes, stride=2, same=False)
        
        self.layer1 = self._make_layer(block, int(self.filters*1), num_blocks[0], stride=2, name='layer1')
        self.layer2 = self._make_layer(block, int(self.filters*2), num_blocks[1], stride=2, name='layer2')
        self.layer3 = self._make_layer(block, int(self.filters*4), num_blocks[2], stride=2, name='layer3')
        self.layer4 = self._make_layer(block, int(self.filters*8), num_blocks[3], stride=2, name='layer4')

        self.fpn0 = upa_block(int(self.filters*4), int(self.filters*4), stride=1, l=1, k=3, same=False, bn=False, cpa=True)
        self.fpn1 = upa_block(int(self.filters*4), int(self.filters*4), stride=1, l=1, k=3, same=False, bn=False, cpa=True)
        self.fpn2 = upa_block(int(self.filters*4), int(self.filters*4), stride=1, l=1, k=3, same=False, bn=False, cpa=True)
        self.fpn3 = upa_block(int(self.filters*4), int(self.filters*4), stride=1, l=1, k=3, same=False, bn=False, cpa=True)
        self.fpn4 = upa_block(int(self.filters*4), int(self.filters*4), stride=1, l=1, k=3, same=False, bn=False, cpa=True)

        
        self.lats = nn.ModuleList([
                upa_block(int(self.filters*2), self.filters, stride=1, l=1, k=1, same=False, bn=False, cpa=True),
                upa_block(int(self.filters*4), self.filters, stride=1, l=1, k=1, same=False, bn=False, cpa=True),
                upa_block(int(self.filters*8), self.filters, stride=1, l=1, k=1, same=False, bn=False, cpa=True),
                upa_block(int(self.filters*16), self.filters, stride=1, l=1, k=1, same=False, bn=False, cpa=True),
                                   ])

        # if self.training:
        self.sem_head = upa_block(int(self.filters*4), num_classes, stride=1, l=1, k=1, bn=False, act=False, cpa=False)

        self.q_m = nn.Linear(int(self.filters*4), int(self.filters*4), bias=False)
        self.k_m = nn.Linear(int(self.filters*4), int(self.filters*4), bias=False)
        self.v_m = nn.Linear(int(self.filters*4), int(self.filters*4), bias=False)

        self.out_mlp = nn.Linear(int(self.filters*4+2), int(self.filters*4), bias=False)
        self.out_mlp1 = nn.Linear(int(self.filters*4), int(self.filters*4), bias=False)
        self.out_mlp2 = nn.Sequential(
              nn.Linear(int(self.filters*4), int(self.filters*8), bias=False),
              nn.ReLU(inplace=True),
              nn.Linear(int(self.filters*8), int(self.filters*4), bias=False),
            )   
        
        self.addcoords = AddCoords()
        
        self.cls_mlp_layer = nn.Linear(int(self.filters*4), int(ar*(num_classes+1)))
        self.box_mlp_layer = nn.Linear(int(self.filters*4), int(ar*4))
        self.coe_mlp_layer = nn.Linear(int(self.filters*4), int(ar*num_classes))
        # self.coe_mlp_layer = nn.Linear(int(self.filters*4), 32)
              
    def _make_layer(self, block, planes, num_blocks, stride, name=None):
        '''w/ upsampling'''
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        self.planes = planes
        planes = planes // num_blocks

        for i, stride in enumerate(strides):

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

        return nn.Sequential(*layers)

    def forward(self, x):
        
        '''Backbone: UPANet V2'''
        
        out0 = self.root(x)

        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        
        # return (out1, out2, out3 ,out4)

        '''FPN: Fully-connected Feature Pyramid'''
        
        outs = [out1, out2, out3, out4] 

        out_68s = []
        out_34s = []
        out_17s = []
        out_09s = []
        out_05s = []
        
        for i in range(len(outs)):
            outs[i] = self.lats[i](outs[i])
            out_68s.append(F.interpolate(outs[i], size=(68 ,68), mode='bilinear', align_corners=False))
            out_34s.append(F.interpolate(outs[i], size=(34, 34), mode='bilinear', align_corners=False))
            out_17s.append(F.interpolate(outs[i], size=(17, 17), mode='bilinear', align_corners=False))
            out_09s.append(F.interpolate(outs[i], size=(9, 9), mode='bilinear', align_corners=False))
            out_05s.append(F.interpolate(outs[i], size=(5, 5), mode='bilinear', align_corners=False))
        
        out_68s = torch.cat(out_68s, 1)
        out_34s = torch.cat(out_34s, 1)
        out_17s = torch.cat(out_17s, 1)
        out_09s = torch.cat(out_09s, 1)
        out_05s = torch.cat(out_05s, 1)
            
        out_05s = self.fpn4(out_05s)
        out_09s = self.fpn3(out_09s)
        out_17s = self.fpn2(out_17s)
        out_34s = self.fpn1(out_34s)
        out_68s = self.fpn0(out_68s)
        
        # return (out_68s, out_34s, out_17s, out_09s, out_05s)

        # if self.training:
        
        sem_out = self.sem_head(out_68s)
        
        ins_out = out_68s

        # return (sem_out, out_68s, out_34s, out_17s, out_09s, out_05s)
        '''Head: Global Attention Head'''
        # addcoords: CoordConv, positional encoding
        # (b, h*w, channel+x_axis+y_axis)
        coord = torch.cat([
            self.addcoords(out_68s).reshape(x.size(0),int(self.filters*4+2),-1),
            self.addcoords(out_34s).reshape(x.size(0),int(self.filters*4+2),-1), 
            self.addcoords(out_17s).reshape(x.size(0),int(self.filters*4+2),-1), 
            self.addcoords(out_09s).reshape(x.size(0),int(self.filters*4+2),-1), 
            self.addcoords(out_05s).reshape(x.size(0),int(self.filters*4+2),-1),
                          ],2) # (b, 258, 6175)
        
        og = self.out_mlp(coord.permute(0,2,1)).permute(0,2,1)  
        # # og = coord
        
        # (q_m, k_m, v_m): assign weight for attention
        # to save time costing: mapping will reduce channel from 258 -> 64
        q = self.q_m(og.permute(0,2,1)).permute(0,2,1) # (B, D, L)
        q = q / (q.shape[1] ** 0.5)
        k = self.k_m(og.permute(0,2,1)).permute(0,2,1)
        v = self.v_m(og.permute(0,2,1)).permute(0,2,1)
        
        # global attention
        # q: (b, 6175, 64), k: (b, 6175, 64), v: (b, 6175, 64)
        # print(q.shape)
        # print(k.shape)
        qk = torch.matmul(q.permute(0,2,1), k) # qT @ k: (b, 6175, 6175)/scale
        qk = torch.softmax(qk, -1)
        # print(qk.shape)
        # print(v.shape)
        # qk = qk / (q.shape[-1] * q.shape[1])
        # qk = qk / (q.shape[-1])
        qkv = torch.matmul(qk, v.permute(0,2,1)).permute(0,2,1)# / q.shape[-1] # qk @ v: (b, 64, 6175)
        qkv  = self.out_mlp1(qkv.permute(0,2,1)).permute(0,2,1)
        
        out = og + qkv
        
        out = out + self.out_mlp2(out.permute(0,2,1)).permute(0,2,1)
        
        cls_mlp_out = self.cls_mlp_layer(out.permute(0,2,1)).reshape(out.size(0), -1, 5)# + class_pred
        box_mlp_out = self.box_mlp_layer(out.permute(0,2,1)).reshape(out.size(0), -1, 4)# + box_pred
        coe_mlp_out = self.coe_mlp_layer(out.permute(0,2,1)).reshape(out.size(0), -1, 4)# + coef_pred
        
        cls_out = cls_mlp_out
        box_out = box_mlp_out
        coe_out = coe_mlp_out

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
    
    net = UPANets(16, 5)
    fpn = net(torch.randn(1, 3, 544, 544))
    for a in fpn:
        print(a.size())
    print('-' * 10)
    print(len(fpn[1:5]))
    # sem_out, ins_out, cls_out, box_out, coe_out = net(torch.randn(1, 3, 544, 544))
    # print(sem_out, ins_out, cls_out, box_out, coe_out)
# test()
