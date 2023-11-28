#import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
#import torch.distributed as dist
#from torch.nn.parallel import DistributedDataParallel as DDP
#from torch.utils.data.distributed import DistributedSampler
import torch.utils.data as data
#from tensorboardX import SummaryWriter
import argparse
import datetime
import re
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('C:/Users/user/Desktop/打包帶走/Yolact_minimal-master')

from utils import timer_mini as timer
from modules.yolact_sod import Yolact
from config import get_config
from utils.coco import COCODetection, train_collate
from utils.common_utils import save_best, save_latest
from eval import evaluate
from fcos import FCOS
from modules.upanets_lite import UPANets
#from torchsummary import summary

#torch.set_default_tensor_type('torch.cuda.FloatTensor')

parser = argparse.ArgumentParser(description='Yolact Training Script')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--cfg', default='res50_custom', help='The configuration name to use.')
parser.add_argument('--train_bs', type=int, default=8, help='total training batch size')
parser.add_argument('--img_size', default=544, type=int, help='The image size for training.')
parser.add_argument('--resume', default=None, type=str, help='The path of the weight file to resume training with.')
parser.add_argument('--val_interval', default=4000, type=int,
                    help='The validation interval during training, pass -1 to disable.')
parser.add_argument('--val_num', default=-1, type=int, help='The number of images for test, set to -1 for all.')
parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
parser.add_argument('--coco_api', action='store_true', help='Whether to use cocoapi to evaluate results.')
parser.add_argument('--og', default=True, help='using og yolact')
parser.add_argument('--total_epoch', default=100, help='total epoch')
parser.add_argument('--upanet', default=True, help='using upanet')
parser.add_argument('--af', default=False, help='anchor free')


# for numpy randomness
# import numpy as np
# np.random.seed(10)
# # for randomness in image augmentation
# import random
# random.seed(10)
# # every PyTorch thing can be fixed with these two lines
# torch.manual_seed(10)
# torch.cuda.manual_seed_all(10)

args = parser.parse_args()
cfg = get_config(args, mode='train')
cfg_name = cfg.__class__.__name__
cfg.og = args.og
cfg.upanet = args.upanet
cfg.af = args.af
cfg.aspect_ratios = [1]
#cfg.cuda = False
#%%
#from modules.upa import UPANets

net = Yolact(cfg)
#net.eval()
#summary(net, (3, 544, 544))
#net = UPANets(cfg, 16, 21, 1, 544)
#backbone = UPANets(64, 21, 1, 544)
#net = FCOS(backbone)
#net.train()

if args.resume:
    assert re.findall(r'res.+_[a-z]+', args.resume)[0] == cfg_name, 'Resume weight is not compatible with current cfg.'
    net.load_weights(cfg.weight, cfg.cuda)
    start_step = int(cfg.weight.split('.pth')[0].split('_')[-1])
else:
#    net.backbone.init_backbone(cfg.weight)
    start_step = 0

dataset = COCODetection(cfg, mode='train')

if 'res' in cfg.__class__.__name__:
    optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
elif cfg.__class__.__name__ == 'swin_tiny_coco':
    optimizer = optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=0.05)
else:
    raise ValueError('Unrecognized cfg.')

train_sampler = None
main_gpu = False
num_gpu = 0
if cfg.cuda:
    cudnn.benchmark = True
    
    cudnn.fastest = True
#    main_gpu = dist.get_rank() == 0
#    num_gpu = dist.get_world_size()
#    net = net.cuda()
    net = net.to('cuda')
#    net = DDP(net.cuda(), [args.local_rank], output_device=args.local_rank, broadcast_buffers=True)
#    train_sampler = DistributedSampler(dataset, shuffle=True)

# shuffle must be False if sampler is specified
#data_loader = data.DataLoader(dataset, cfg.bs_per_gpu, num_workers=cfg.bs_per_gpu // 2, shuffle=(train_sampler is None),
#                              collate_fn=train_collate, pin_memory=False, sampler=train_sampler)
data_loader = data.DataLoader(dataset, cfg.train_bs, num_workers=0, shuffle=True,
                           collate_fn=train_collate, pin_memory=False)

epoch_seed = 0
map_tables = []
training = True
timer.reset()
step = start_step
val_step = start_step
epoch_iter = len(dataset) // args.train_bs
num_epochs = math.ceil(cfg.lr_steps[-1] / epoch_iter)
#writer = SummaryWriter(f'tensorboard_log/{cfg_name}')
print(f'Number of all parameters: {sum([p.numel() for p in net.parameters()])}\n')

#%%
#if main_gpu:
    
#print(f'Number of all parameters: {sum([p.numel() for p in net.parameters()])}\n')

time_last = 0
try:  # try-except can shut down all processes after Ctrl + C.
#    train_loss_list=[]
#    box
    losses_c = []
    losses_b = []
    losses_s = []
    losses_i = []
    losses_m = []
    losses_total = []
    itrs = []

    test_losses_c = []
    test_losses_b = []
    test_losses_s = []
    test_losses_i = []
    test_losses_m = []
    test_losses_total = []
    test_itrs = []
    test_itr = 0
    for epoch in range(args.total_epoch):
#        if train_sampler:
#            epoch_seed += 1
#            train_sampler.set_epoch(epoch_seed)

        for images, targets, masks in data_loader:
            timer.start()
            if cfg.warmup_until > 0 and step <= cfg.warmup_until:  # warm up learning rate.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = (cfg.lr - cfg.warmup_init) * (step / cfg.warmup_until) + cfg.warmup_init

            if step in cfg.lr_steps:  # learning rate decay.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cfg.lr * 0.1 ** cfg.lr_steps.index(step)

            if cfg.cuda:
#                images = images.cuda().detach()
#                targets = [ann.cuda().detach() for ann in targets]
#                masks = [mask.cuda().detach() for mask in masks]

                images = images.to('cuda').detach()
                targets = [ann.to('cuda').detach() for ann in targets]
                masks = [mask.to('cuda').detach() for mask in masks]

            with timer.counter('for+loss'):
                loss_c, loss_b, loss_m, loss_s = net(images, targets, masks)
#                _, losses = net(images, targets=targets, masks=masks)
                
#                loss_c = losses['loss_cls']
#                loss_b = losses['loss_box']
#                loss_s = losses['loss_seg'] + losses['loss_ins']
#                loss_m = losses['loss_center']

                losses_c.append(loss_c.item())
                losses_b.append(loss_b.item())
                losses_s.append(loss_s.item())
#                losses_i.append(losses['loss_ins'].item())
                losses_m.append(loss_m.item())
                itrs.append(step)
                
#                if cfg.cuda:
#                    # use .all_reduce() to get the summed loss from all GPUs
#                    all_loss = torch.stack([loss_c, loss_b, loss_m, loss_s], dim=0)
#                    dist.all_reduce(all_loss)

            with timer.counter('backward'):
                loss_total = loss_c + loss_b + loss_s + loss_m
                losses_total.append(loss_total.item())
#                optimizer.zero_grad()
#                loss_total.backward()
                loss = loss_total#/8
                loss.backward()
                
            with timer.counter('update'):
#                if step % 7==0:
                optimizer.step()
                optimizer.zero_grad()

            time_this = time.time()
            if step > start_step:
                batch_time = time_this - time_last
                timer.add_batch_time(batch_time)
                time_last = time_this
            else:
                time_last = time_this

            if step % 100 == 0 and step != start_step:
#                if (not cfg.cuda) or main_gpu:
#                    cur_lr = optimizer.param_groups[0]['lr']
                time_name = ['batch', 'data', 'for+loss', 'backward', 'update']
                t_t, t_d, t_fl, t_b, t_u = timer.get_times(time_name)
                seconds = (cfg.lr_steps[-1] - step) * t_t
                eta = str(datetime.timedelta(seconds=seconds)).split('.')[0]
        
                        # Get the mean loss across all GPUS for printing, seems need to call .item(), not sure
                l_c = loss_c.item()
                l_b = loss_b.item()
#                if cfg.og == True:
#                    l_m = loss_m.item()
#                else:
#                    l_m = 0
                l_s = loss_s.item()
                l_m = loss_m.item()
        
        #                    writer.add_scalar('loss/class', l_c, global_step=step)
        #                    writer.add_scalar('loss/box', l_b, global_step=step)
        #                    writer.add_scalar('loss/mask', l_m, global_step=step)
        #                    writer.add_scalar('loss/semantic', l_s, global_step=step)
        #                    writer.add_scalar('loss/total', loss_total, global_step=step)
                cur_lr = optimizer.param_groups[0]['lr']
                print(f'step: {step} | lr: {cur_lr:.2e} | l_class: {l_c:.3f} | l_box: {l_b:.3f} | '
                      f'l_mask: {l_m:.3f} | l_semantic: {l_s:.3f} | t_t: {t_t:.3f} | t_d: {t_d:.3f} | '
                      f't_fl: {t_fl:.3f} | t_b: {t_b:.3f} | t_u: {t_u:.3f} | ETA: {eta}')
#                break
#            if args.val_interval > 0 and step % args.val_interval == 0 and step != start_step:
#                if (not cfg.cuda) or main_gpu:

#            if ((not cfg.cuda) or main_gpu) and step == val_step + 1:
#                timer.start()  # the first iteration after validation should not be included
            timer.reset()
            step += 1
            
        plt.plot(itrs, losses_total, c='gray', label='loss_t')
        plt.plot(itrs, losses_c, alpha=0.5, c='green', label='loss_c')
        plt.plot(itrs, losses_b, alpha=0.5, c='blue', label='loss_b')
        plt.plot(itrs, losses_s, alpha=0.5, c='red', label='loss_s')
#        plt.plot(itrs, losses_i, alpha=0.5, c='purple', label='loss_i')
        plt.plot(itrs, losses_m, alpha=0.5, c='orange', label='loss_m')
        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.title('train')
#        plt.xticks(np.arange(0, itrs[-1], step=len(itrs)/(epoch+1))) 
        plt.legend(loc=0)  
        plt.grid()
        plt.show()
#        with torch.no_grad():
            
        if epoch % 2 == 0:
            
            test_itr = test_itr +1
            val_step = step
            net.eval()
            table, box_row, mask_row = evaluate(net, cfg, step)
            map_tables.append(table)
            net.train()

#            test_losses_c = test_losses_c + losses_dict['losses_c']
#            test_losses_b = test_losses_c + losses_dict['losses_c']
#            test_losses_s = test_losses_c + losses_dict['losses_c']
#            test_losses_i = test_losses_c + losses_dict['losses_c']
#            test_losses_m = test_losses_c + losses_dict['losses_c']
#            test_losses_total = test_losses_c + losses_dict['losses_c']
#            test_itrs = test_itrs + losses_dict['itrs']
#           
#            plt.plot(test_itrs, test_losses_total, c='gray', label='loss_t')
#            plt.plot(test_itrs, test_losses_c, alpha=0.5, c='green', label='loss_c')
#            plt.plot(test_itrs, test_losses_b, alpha=0.5, c='blue', label='loss_b')
#            plt.plot(test_itrs, test_losses_s, alpha=0.5, c='red', label='loss_s')
#            plt.plot(test_itrs, test_losses_i, alpha=0.5, c='purple', label='loss_i')
#            plt.plot(test_itrs, test_losses_m, alpha=0.5, c='green', label='loss_m')
#            plt.xlabel('iter')
#            plt.ylabel('loss')
#            plt.title('test')
#            plt.xticks(np.arange(0, test_itrs[-1], step=len(test_itrs)/(test_itr))) 
#            plt.legend(loc=0)
#            plt.grid()
#            plt.show()
            
#            timer.reset()  # training timer and val timer share the same Obj, so reset it to avoid conflict
        
        
#            writer.add_scalar('mAP/box_map', box_row[1], global_step=step)
#            writer.add_scalar('mAP/mask_map', mask_row[1], global_step=step)

#        save_best(net if cfg.cuda else net, mask_row[1], cfg_name, step)
    net.eval()
    table, box_row, mask_row = evaluate(net, cfg, step)
    map_tables.append(table)    
#        if step >= cfg.lr_steps[-1]:
#            training = False
#
#            if (not cfg.cuda) or main_gpu:
#                save_latest(net if cfg.cuda else net, cfg_name, step)
#
#                print('\nValidation results during training:\n')
#                for table in map_tables:
#                    print(table, '\n')
#
#                print('Training completed.')
#
#            break

except KeyboardInterrupt:
    if (not cfg.cuda) or main_gpu:
        save_latest(net if cfg.cuda else net, cfg_name, step)

        print('\nValidation results during training:\n')
        for table in map_tables:
            print(table, '\n')
