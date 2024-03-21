import torch
import numpy as np
import cv2
from adapt_sd import StableDiffusion
from nnutils.pose import Poser
import os
from tqdm import tqdm
import torchvision
img_path = 'database/DAVIS/JPEGImages/Full-Resolution/cat-pikachiu05/'
mask_path = 'database/DAVIS/Annotations/Full-Resolution/cat-pikachiu05/'


model = StableDiffusion(variant="objaverse",v2_highres=False,prompt="a cat",im_path="data.png",scale=5.,precision='autocast')

def img2embed(img):
    with torch.no_grad():
        tforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop((256, 256))
        ])

        input_im = tforms(img)
        # get input embedding
        clip_emb = model.model.get_learned_conditioning(input_im.float()).tile(1,1,1).detach()
        vae_emb = model.model.encode_first_stage(input_im.float()).mode().detach()
    return clip_emb,vae_emb

# import pdb;pdb.set_trace()
for i in tqdm(range(82)):
    img = cv2.imread(img_path+'%05d.jpg'%i)/255.
    mask = cv2.imread(mask_path+'%05d.jpg'%i)/255.
    gt_image = torch.from_numpy(img[:,:,[2,1,0]]).squeeze().cuda().float().moveaxis(-1,0)
    gt_mask = torch.from_numpy(mask[:,:,0]).squeeze().cuda().float()
    background = torch.tensor([1.,1.,1.],requires_grad=False).cuda()
    gt_image = gt_image*gt_mask+(1.-gt_mask)*(background[...,None,None])
    input_img = torch.cat([torch.ones([3,420,1920]).cuda(),gt_image,torch.ones([3,420,1920]).cuda()],dim=1)
    input_img = input_img[None,...]* 2. - 1.
    # input_img = torch.cat([torch.ones([420,1920,3]).cuda(),input_img,torch.ones([420,1920,3]).cuda()])
    clip_emb, vae_emb = img2embed(input_img)
    clip_emb = clip_emb.detach().cpu().numpy()
    vae_emb = vae_emb.detach().cpu().numpy()
    np.save(img_path+'%05d_clip.npy'%i,clip_emb)
    np.save(img_path+'%05d_vae.npy'%i,vae_emb)
    
    
    