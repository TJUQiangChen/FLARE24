import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

def pad(im):
    shape=im.shape
    cx,cy= int((shape[0])/2),int(shape[1]/2)
    shape=im.shape
    ss=[[256-cx,256+shape[0]-cx],[256-cy,256+shape[1]-cy]]
    empty1=np.zeros([512,512,im.shape[-1]])
    empty1[ss[0][0]:ss[0][1],ss[1][0]:ss[1][1]]=im
    return empty1

'''
MR Prepare: 3D to 2D slice (image)
'''

# Need to preprocess AMOS and LLD datasets

if not os.path.exists('datasets/MR_2d'):    
    os.mkdir('datasets/MR_2d')
    
# AMOS
root=r'/mnt16t_ext/dinghaoyu/FLARE24/datasets/MR/Training/AMOS_MR_good_spacing-833'

    
for d in os.listdir(root)[0:250]:
    im_list=[]
    img = nib.load(os.path.join(root,d))
    try:
        image_data=img.get_fdata().transpose((1,0,2))[::-1]
    except EOFError:
        continue

    image_data = image_data.copy()
    origin_shape=image_data.shape
    target_shape= [int(origin_shape[0]*img.header['pixdim'][1]),int(origin_shape[1]*img.header['pixdim'][2]), int(origin_shape[2]*img.header['pixdim'][3]/4)]
    try:
        image_data=torch.from_numpy(image_data).unsqueeze(0).unsqueeze(0)
        image_data=F.interpolate(image_data,target_shape,mode='trilinear').numpy()[0,0]
        image_data=pad(image_data)
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
    except ValueError:
        continue
    
    for i in range(image_data.shape[2]-1):
        name = d[:10] + '{}_'.format(i) + d[10:]
        np.save('datasets/MR_2d/{}'.format(name.replace('.nii.gz','.npy')),image_data[:,:,i:i+1])
    print('AMOS finish:', name)

# LLD
root=r'/mnt16t_ext/dinghaoyu/FLARE24/datasets/MR/Training/LLD-MMRI-3984'
for d in os.listdir(root)[0:250]:
    im_list=[]
    img = nib.load(os.path.join(root,d))
    try:
        image_data=img.get_fdata().transpose((1,0,2))[::-1]
    except EOFError:
        continue

    image_data = image_data.copy()
    origin_shape=image_data.shape
    target_shape= [int(origin_shape[0]*img.header['pixdim'][1]),int(origin_shape[1]*img.header['pixdim'][2]), int(origin_shape[2]*img.header['pixdim'][3]/4)]
    try:
        image_data=torch.from_numpy(image_data).unsqueeze(0).unsqueeze(0)
        image_data=F.interpolate(image_data,target_shape,mode='trilinear').numpy()[0,0]
        image_data=pad(image_data)
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
    except ValueError:
        continue
    
    for i in range(image_data.shape[2]-1):
        name = d[:10] + '{}_'.format(i) + d[10:]
        np.save('datasets/MR_2d/{}'.format(name.replace('.nii.gz','.npy')), image_data[:,:,i:i+1])
    print('LLD finish:', name)




