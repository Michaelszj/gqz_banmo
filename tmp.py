import numpy as np
import torch
import cv2
def rtk_from_angles(delta, phi, resolution=256, radius=1.):
    rtk_novel = torch.zeros([4,4],dtype=torch.float32).cuda()
    Rmat = torch.tensor([[-np.sin(delta),0.,-np.cos(delta)],
                            [-np.cos(delta)*np.sin(phi),-np.cos(phi),np.sin(delta)*np.sin(phi)],
                            [-np.cos(delta)*np.cos(phi),np.sin(phi),np.sin(delta)*np.cos(phi)]])
    Tmat = torch.tensor([0.,0.,1.])*radius
    r = float(resolution)
    rtk_novel[3] = torch.tensor([r,r,r/2,r/2]).cuda()
    rtk_novel[:3,:3] = Rmat
    rtk_novel[:3,3]  = Tmat
    return rtk_novel
    
def angles_from_rtk(rtk):
    Rmat = rtk[:3,:3]
    z = rtk[2]
    phi = torch.arcsin(z[1])
    cos_phi = torch.cos(phi)
    theta = torch.arccos(-z[0]/cos_phi)
    if z[2]<0:
        theta = -theta
    return theta, phi
        
cat = cv2.imread('database/DAVIS/JPEGImages/Full-Resolution/cat-pikachiu05/00038.jpg')
mask = cv2.imread('database/DAVIS/Annotations/Full-Resolution/cat-pikachiu05/00038.jpg')[:,:,:1]
rgba = np.concatenate([cat,mask],axis=-1)
rgba = np.concatenate([np.zeros([420,1920,4],dtype=np.uint8),rgba,np.zeros([420,1920,4],dtype=np.uint8)],axis=0)
cv2.imwrite('cat_new.png',rgba)