import numpy as np
import torch
import cv2
from tqdm import tqdm
import os
path = 'database/DAVIS/Annotations/Full-Resolution/camel'


def mask_to_sdf(mask):
    dup   = mask[:-1]-mask[1:]
    dleft = mask[:,:-1]-mask[:,1:]
    
    down_border = torch.cat([dup,torch.zeros_like(mask[:1]).cuda()],dim=0)
    up_border = torch.cat([torch.zeros_like(mask[:1]).cuda(),dup],dim=0)
    right_border = torch.cat([dleft,torch.zeros_like(mask[:,:1]).cuda()],dim=1)
    left_border = torch.cat([torch.zeros_like(mask[:,:1]).cuda(),dleft],dim=1)
    
    border = torch.logical_or(torch.logical_or(down_border!=0,up_border!=0),torch.logical_or(right_border!=0,left_border!=0))
    border_points = torch.nonzero(border).unsqueeze(1)
    # border_points_2 = border_points**2
    pixel_y,pixel_x = torch.meshgrid(torch.arange(1080),torch.arange(1920))
    pixels = torch.stack([pixel_y,pixel_x],dim=-1).cuda().float()
    distance = torch.zeros_like(pixels[:,:,0]).cuda()
    for i in range(1080):
        distances = torch.sqrt(torch.sum((border_points-pixels[i:i+1])**2,dim=-1))
        distance[i] = torch.min(distances,dim=0)[0]
    # import pdb;pdb.set_trace()
    return distance+0.5

for root, dirs, filenames in os.walk(path):
    for filename in filenames:
        if (filename[-4:]!='.png') or (filename[:3]=='vis'):
            continue
        mask = cv2.imread(os.path.join(root, filename))
        mask = torch.from_numpy(mask)[:,:,0].float().cuda()
        mask[mask>=128.]=255.
        mask[mask<128.] =0.
        sdf = mask_to_sdf(mask)
        # import pdb;pdb.set_trace()
        save_path = os.path.join(root, filename[:-4]+'.npy')
        np.save(save_path,sdf.cpu().numpy())
        print('writing mask:'+save_path)