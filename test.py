import torch
import pytorch3d.transforms as T
from tqdm import tqdm
import numpy as np

bs=10
for i in tqdm(range(bs)):
    angle = np.pi*2*i/bs
    Rmat = torch.inverse(torch.tensor([[np.sin(angle),0.,np.cos(angle)],
                                        [0.,-1.,0.],
                                        [-np.cos(angle),0.,np.sin(angle)]]))
    print(Rmat)
    print(T.matrix_to_quaternion(Rmat))