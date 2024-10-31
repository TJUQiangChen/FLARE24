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
CT Prepare: 3D to 2D slice (image and mask)
'''
# modify this
root=r'/mnt16t_ext/dinghaoyu/FLARE24/datasets/CT/CT_image'
label_root=r'/mnt16t_ext/dinghaoyu/FLARE24/datasets/CT/CT_label'


# os.chdir('Style_Translation')

if not os.path.exists('datasets/CT_2d'):
    os.mkdir('datasets/CT_2d')

if not os.path.exists('datasets/CT_2d_label'):    
    os.mkdir('datasets/CT_2d_label')


A_imgs=[]
for f in os.listdir(root):
    img = nib.load(os.path.join(root,f))
    try:
        image_data=img.get_fdata().transpose((1,0,2))[::-1]
    except EOFError:
        continue

    mask_data=nib.load(os.path.join(label_root, f[: -12] +'.nii.gz')).get_fdata().transpose((1,0,2))[::-1]
    seg = sitk.ReadImage(os.path.join(label_root, f[: -12] +'.nii.gz'))
    seg_array = sitk.GetArrayFromImage(seg)
    z = np.any(seg_array, axis=(1, 2))
    start_slice, end_slice = np.where(z)[0][[0, -1]]
    start_slice=max(start_slice-2,0)
    end_slice=min(end_slice+2,image_data.shape[-1])
    image_data=image_data[:,:,start_slice:end_slice+1]
    mask_data=mask_data[:,:,start_slice:end_slice+1]

    image_data[image_data>350]=350
    image_data[image_data<-350]=-350
    image_data= (np.clip(image_data,-350,350)+350)/700
    origin_shape=image_data.shape
    target_shape= [int(origin_shape[0]*img.header['pixdim'][1]),int(origin_shape[1]*img.header['pixdim'][2]), int(origin_shape[2]*img.header['pixdim'][3]/4)]

    image_data=torch.from_numpy(image_data).unsqueeze(0).unsqueeze(0)
    image_data=F.interpolate(image_data,target_shape, mode = "trilinear").numpy()[0,0]
    try:
        mask_data=torch.from_numpy(mask_data.copy()).unsqueeze(0).unsqueeze(0)
        mask_data=F.interpolate(mask_data,target_shape,mode='nearest').numpy()[0,0]
        mask_data=pad(mask_data)

    except ValueError:
        continue
    
    for i in range(image_data.shape[2]-1):
        name = f[:10] + '_{}'.format(i) + f[10:]
        np.save('datasets/CT_2d/{}'.format(name.replace('.nii.gz','.npy')),image_data[:,:,i:i+1])
        np.save('datasets/CT_2d_label/{}'.format(name.replace('img','label').replace('.nii.gz','.npy')),mask_data[:,:,i:i+1])
    print('finish:', name)
