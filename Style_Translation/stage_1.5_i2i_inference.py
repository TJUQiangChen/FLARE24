import os
import torch
from sklearn.cluster import KMeans
import numpy as np
from i2i_solver import i2iSolver
import random
import argparse
import SimpleITK as sitk
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# python stage_1.5_i2i_inference.py --source_npy_dirpath /mnt16t_ext/dinghaoyu/FLARE24/Style_Translation/datasets/CT_2d --target_npy_dirpath /mnt16t_ext/dinghaoyu/FLARE24/Style_Translation/datasets/MR_2d --save_dirpath /mnt16t_ext/dinghaoyu/FLARE24/Style_Translation/datasets/result

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, default='/mnt16t_ext/dinghaoyu/FLARE24/Style_Translation/checkpoints_dir/enc_0200.pt')
parser.add_argument('--source_npy_dirpath', type=str, default='/mnt16t_ext/lijiaxi/STYLE_DATASET/DAR_CT_npy')
parser.add_argument('--target_npy_dirpath', type=str, default='/mnt16t_ext/lijiaxi/STYLE_DATASET/DAR_MR_npy')
parser.add_argument('--save_dirpath', type=str, default='datasets/source2target_training_npy')
parser.add_argument('--k_means_clusters', type=int, default=6)

opts = parser.parse_args()
trainer=i2iSolver(None)
state_dict = torch.load(opts.ckpt_path)
trainer.enc_c.load_state_dict(state_dict['enc_c'])
trainer.enc_s_a.load_state_dict(state_dict['enc_s_a'])
trainer.enc_s_b.load_state_dict(state_dict['enc_s_b'])
trainer.dec.load_state_dict(state_dict['dec'])
trainer.cuda()

styles=[]
for f2 in os.listdir(opts.target_npy_dirpath)[0:500]:
    if 'label' not in f2:
        imgs = np.load(os.path.join(opts.target_npy_dirpath, f2))
        for i in range(int(imgs.shape[-1]/6),int(imgs.shape[-1]/6*5)):
            img = imgs[:, :, i]
            with torch.no_grad():
                single_img = torch.from_numpy((img * 2 - 1)).unsqueeze(0).unsqueeze(0).cuda().float()
                s=trainer.enc_s_b(single_img).cpu().numpy()[0]
                styles.append(s)
n_clusters=opts.k_means_clusters
k_mean_results = KMeans(n_clusters=opts.k_means_clusters, random_state=9).fit_predict(styles)

target_images = os.listdir(opts.target_npy_dirpath)
for f in os.listdir(opts.source_npy_dirpath):
    imgs = np.load(os.path.join(opts.source_npy_dirpath, f))
    idx = random.randint(0,len(target_images)-1)
    target_img = np.load(os.path.join(opts.target_npy_dirpath, target_images[idx]))
    target_img = target_img[:, :, int(target_img.shape[-1]/2)]
    with torch.no_grad():
        single_img = torch.from_numpy((target_img * 2 - 1)).unsqueeze(0).unsqueeze(0).cuda().float()
        s=trainer.enc_s_a(single_img)[0].unsqueeze(0)
    nimgs = np.zeros_like(imgs, dtype=np.float32)
    for i in range(imgs.shape[-1]):
        img = imgs[:, :, i]
        single_img = torch.from_numpy((img * 2 - 1)).unsqueeze(0).unsqueeze(0).cuda().float()
        transfered_img = trainer.inference(single_img, s)
        transfered_img = (((transfered_img + 1) / 2).cpu().numpy()).astype(np.float32)[0, 0]
        nimgs[:, :, i] = transfered_img
    nimgs = nimgs.transpose(2,0,1)
    img = sitk.GetImageFromArray(nimgs)
    sitk.WriteImage(img, os.path.join(opts.save_nii_dirpath, f.replace('.npy', '.nii.gz')) )
    print('End processing %s.' % f)



