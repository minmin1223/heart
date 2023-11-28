'''UPANets in PyTorch.

Processing model in cifar10 by Ching-Hsun Tseng and Jia-Nan Feng
'''
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import sys
sys.path.append('E:/hank/Res2NET-PoolNet-master/networks/')
#import resize_right
from torchvision import transforms

class upa_block(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1, cat=False, same=False, w=2, l=2, up=False):
        
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
            out = F.layer_norm(out, out.size()[1:])
            
        else:
            out = F.layer_norm(out, out.size()[1:])
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
            
#            if self.up == False:               
            out =   self.avgpool(out)# + self.maxpool(out)
#                out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=True)
                
        return out
   
class SPA(nn.Module):
    '''Spatial Pixel Attention'''

    def __init__(self, img, out=1):
        
        super(SPA, self).__init__()
        
        self.sp_ffc = nn.Sequential(
            nn.Linear(img, out)
            )   
#        self.sp_ffc = nn.Conv2d(img**2, out**2, kernel_size=1, bias=False)
        
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
#        filter_nums = filter_nums*2
        self.in_planes = filter_nums
        self.filters = filter_nums
        print('filters based:', self.filters)
        self.img = img
        w = 2
        
        self.root = nn.Sequential(
                nn.Conv2d(3, int(self.in_planes*w), kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(int(self.in_planes*w)),
                nn.ReLU(),
                nn.Conv2d(int(self.in_planes*w), self.in_planes*1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.in_planes),
                nn.ReLU(),
                )
        
        self.emb = CPA(3, self.in_planes, stride=2, same=True)
        
        self.layer1 = self._make_layer(block, int(self.filters*1), num_blocks[0], 2, name='layer1')
        self.layer2 = self._make_layer(block, int(self.filters*2), num_blocks[1], 2, name='layer2')
        self.layer3 = self._make_layer(block, int(self.filters*4), num_blocks[2], 2, name='layer3')
        self.layer4 = self._make_layer(block, int(self.filters*8), num_blocks[3],2, name='layer4')
#        
#        self.layer4_up = self._make_layer(block, int(self.filters*4), num_blocks[3], 1, up=True, name='layer4_up')
#        self.layer3_up = self._make_layer(block, int(self.filters*2), num_blocks[2], 2, up=True, name='layer3_up')
#        self.layer2_up = self._make_layer(block, int(self.filters*1), num_blocks[1], 2, up=True, name='layer2_up')
#        self.layer1_up = self._make_layer(block, int(self.filters*0.5), num_blocks[0], 2, up=True, name='layer1_up')
        
        self.root_up = nn.Sequential(
                nn.Conv2d(int(self.filters*31), int(self.filters*1), kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(int(self.filters*1)),
                nn.ReLU(),
                nn.Conv2d(int(self.filters*1), num_classes, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_classes),
                nn.ReLU(),
                )
        self.emb_up = CPA(int(self.filters*31), num_classes, stride=1, up=False, same=True)
#        self.spa = SPA(int(112*1)**2, int(112*1)**2)
#        self.root_out = nn.Sequential(
#                nn.Conv2d(int(self.filters*1), int(self.filters*2), kernel_size=3, padding=1, bias=False),
#                nn.BatchNorm2d(int(self.filters*2)),
#                nn.ReLU(),
#                nn.Conv2d(int(self.filters*2), 1, kernel_size=3, padding=1, bias=False),
#                nn.BatchNorm2d(1),
#                nn.ReLU(),
#                )
#        self.emb_out = CPA(int(self.filters*1), 1, stride=2, up=True, same=False)
        
        
    def _make_layer(self, block, planes, num_blocks, stride, up=False, name=None):
        
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        self.planes = planes
        planes = planes // num_blocks
#        print('='*10)
#        print(name)
        for i, stride in enumerate(strides):
#            print(i, stride)
#            print('in:', self.in_planes)
            
            if up == True:
                if i != 0 and stride == 1:
                   
                    layers.append(block(self.in_planes, planes, stride, cat=True, up=up))                
                    self.in_planes = self.in_planes + planes 
                        
                else:   
                    if name != 'layer4_up':
#                        print('in dense:', self.in_planes * 2)
                        self.in_planes = self.in_planes * 2 
                        
                    layers.append(block(self.in_planes, self.planes, stride, up=up))
                    strides.append(1)
                    self.in_planes = self.planes
            
            else:
                if i == 0 and stride == 1:
                    layers.append(block(self.planes, self.planes, stride, same=True, up=up))
                    strides.append(1)
                    self.in_planes = self.planes
                    
                elif i != 0 and stride == 1:
                    layers.append(block(self.in_planes, planes, stride, cat=True, up=up))                
                    self.in_planes = self.in_planes + planes 
                        
                else:   
                    layers.append(block(self.in_planes, self.planes, stride, up=up))
                    strides.append(1)
                    self.in_planes = self.planes
            
#            print('out:', self.in_planes)
        return nn.Sequential(*layers)

    def forward(self, x):
        
#        _, _, self.img, _ = x.shape
        
#        og_img = x.shape[-2:]
        in_x = x
#        in_x = F.avg_pool2d(in_x, 2)
#        in_x = F.interpolate(in_x, size=(224,224), mode='bilinear', align_corners=True) # change
#        x = self.dw(x)
#        self.img = x.shape[-2:]
#        x = F.avg_pool2d(x, 2)
#        print('in: ',in_x.shape)
#        self.img = in_x.shape[-2:]
#        in_x = F.interpolate(in_x, scale_factor=2, mode='bilinear', align_corners=True)
#        print('in: ',in_x.shape)
        self.img = in_x.shape[-2:]
#        in_x = torch.cat([x, og], 1)
        out0 = self.root(in_x)
#        out0 = F.avg_pool2d(out0, 2)
        out0 = self.emb(in_x, out0)
#        out0 = F.interpolate(out0, scale_factor=0.5, mode='bilinear', align_corners=True)
        out0 = F.avg_pool2d(out0, 2)
#        print('x:', x.shape)
#        print('out0:', out0.shape)
#        out0_m = out0
#        self.img = out0.shape[-2:]
#        out0 = F.avg_pool2d(out0, 2)
#        out0_m = F.max_pool2d(out0, kernel_size=3, stride=2, padding=1) # change
#        self.img = out0_m.shape[-2:]
    
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
#                
##        print('out0 max:', out0_m.shape)
#        print('out0:', out0.shape)
#        print('out1:', out1.shape)
#        print('out2:', out2.shape)
#        print('out3:', out3.shape)
#        print('out4:', out4.shape)
#                
#        out4_up = self.layer4_up(out4)
##        print('up out4:', out4_up.shape)
#        out4_up, out3 = size_check(out4_up, out3)
#        out3_up = self.layer3_up(torch.cat([out4_up, out3], 1))
#        out3_up, out2 = size_check(out3_up, out2)
#        out2_up = self.layer2_up(torch.cat([out3_up, out2], 1))
#        out2_up, out1 = size_check(out2_up, out1)
#        out1_up = self.layer1_up(torch.cat([out2_up, out1], 1))
#                
        out0 = F.interpolate(out0, size=self.img, mode='bilinear', align_corners=False)
        out1 = F.interpolate(out1, size=self.img, mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, size=self.img, mode='bilinear', align_corners=False)
        out3 = F.interpolate(out3, size=self.img, mode='bilinear', align_corners=False)
        out4 = F.interpolate(out4, size=self.img, mode='bilinear', align_corners=False)
#    
#        print('up out4:', out4_up.shape)
#        print('up out3:', out3_up.shape)
#        print('up out2:', out2_up.shape)
#        print('up out1:', out1_up.shape)
##        
#        out4_up = F.interpolate(out4_up, size=self.img, mode='bilinear', align_corners=True)
#        out3_up = F.interpolate(out3_up, size=self.img, mode='bilinear', align_corners=True)
#        out2_up = F.interpolate(out2_up, size=self.img, mode='bilinear', align_corners=True)
#        out1_up = F.interpolate(out1_up, size=self.img, mode='bilinear', align_corners=True)
            
        out = torch.cat([
                out0, 
                out1, out2, out3, out4,# x,
#                out1_up,
#                out2_up, out3_up, out4_up
                ], 1)
    
#        spa_out = F.interpolate(torch.mean(out, 1, keepdim=True), size=(int(112*1),int(112*1)), mode='bilinear', align_corners=True) # change
#        spa_out = self.spa(spa_out)
#        spa_out = F.interpolate(spa_out, size=out.shape[-2:], mode='bilinear', align_corners=True) # change
#        out = F.layer_norm(out + spa_out, out.size()[1:])
#        print('in emb_out: ',out.shape)
        out0_up = self.root_up(out)
        out = self.emb_up(out, out0_up)
#        out = out + x
#        out = out# + spa_out
        
#        out = out + spa_out
#        print('emb_out: ',out.shape)
        
#        out_ = self.root_out(out)
#        out = self.emb_out(out, out_)
#        print('out: ',out.shape)
#        if out.shape[-2:] != x.shape[-2:]:
#            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=True)
                
#        out = F.upsample(out, scale_factor=2, mode='bilinear', align_corners=True)
#        out, _ = size_check(out, x, True)
#        out = F.interpolate(out, size=og_img, mode='bilinear')
#        print('out: ',out.shape)
#        
        return out

def size_check(x1, x2, label=False):
    
#    if x1.shape != x2.shape:
#        if label == True and int(x1.shape[-2] * x1.shape[-1]) != int(x2.shape[-2] * x2.shape[-1]):
#            x1 = F.interpolate(x1, size=x2.shape[-2:])
#           
#        else:
    if x1.shape[2:] != x2.shape[2:]:
        if int(x1.shape[-2] * x1.shape[-1]) > int(x2.shape[-2] * x2.shape[-1]):
            x2 = F.interpolate(x2, size=x1.shape[-2:], mode='bilinear', align_corners=True)
            
        else:
            x1 = F.interpolate(x1, size=x2.shape[-2:], mode='bilinear', align_corners=True)
            
    return x1, x2

    
def init_weights(m):

    if isinstance(m, nn.Linear) or type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)

    if isinstance(m, nn.Conv2d) or type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight, 0, 0.01)

    if isinstance(m, nn.Parameter) or type(m) == nn.Parameter:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        
#    if isinstance(m, nn.BatchNorm2d) or type(m) == nn.BatchNorm2d:
#        torch.nn.init.normal_(m.weight, 0, 0.01)
    try:
        m.bias.data.zero_()
    except:
        pass

def UPANets(f=64, c=100, block=1, img=32):
    
    return upanets(upa_block, [int(4*block), int(4*block), int(4*block), int(4*block)], f, num_classes=c, img=img)

def test():
    
    net = UPANets(64, c=20)
    out = net(torch.randn(1, 3, 544, 544))
#    print(y.size())
    return out

#out = test()
#from torchsummary import summary
#net = upanets_locate(32, 64)
#summary(net, (3, 32, 32))

#if __name__ == '__main__':
#    #images = torch.rand(2, 3, 224, 224)
#    images = torch.rand(1, 3, 264, 264).cuda(0)
#    model = UPANets()
#    model = model.cuda(0)    
#    total = sum([param.nelement() for param in model.parameters()])
#    print('  + Number of params: %.4fM' % (total / 1e6))
#    print(model(images).size())
#    print('Memory useage: %.4fM' % ( torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))
    
    
    