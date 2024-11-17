# Style Translation
The codes for the work "A 3D Unsupervised Domain Adaption Framework Combining Style Translation and Self-Training for Abdominal Organs Segmentation".

## 1. Prepare data

## Prerequisites
1. Respacing the source and target domain images (both should be gray) with the same XY plane resolutions, and crop/pad to the size of [512, 512, d] in terms of [width, height, depth].

2. Normalize each 3D images to [0, 1], and extract 2D slices from 3D volumes along depth-axis.

3. Stack the list of 2D slices at zero dimension for the two domains respectively, resulting in 3D tensor with size of [N, 512, 512].

## Prepare Style Translation Training Samples
```bash
# Prepare CT training data
python prepare_data_for_abdominal_CT_2d.py \
    --data_root '../datasets/CT/CT_image' \
    --label_root '../datasets/CT/CT_label' \
    --save_path '../datasets/CT/ST_CT_2d'

# Prepare MR training data
python prepare_data_for_abdominal_MR_2d.py \
    --AMOS_root '../datasets/MR/Training/AMOS_MR_good_spacing-833' \
    --LLD_root '../datasets/MR/Training/LLD-MMRI-3984' \
    --save_path '../datasets/MR/Training/ST_MR_2d'
```

## Prepare Style Translation Inference Samples
```bash
# Prepare CT inference data
python prepare_data_for_abdominal_CT_3d.py \
    --data_root '../datasets/CT/CT_image' \
    --label_root '../datasets/CT/CT_label' \
    --save_img_path '../datasets/CT/ST_CT_3d/CT_3d_img_npy' \
    --save_lb_path '../datasets/CT/CT2MR_label'

# Prepare MR inference data
python prepare_data_for_abdominal_MR_3d.py \
    --AMOS_root '../datasets/MR/Training/AMOS_MR_good_spacing-833' \
    --LLD_root '../datasets/MR/Training/LLD-MMRI-3984' \
    --save_path '../datasets/MR/Training/ST_MR_3d'
```

## Finally, CT to MR Dataset save path
```bash
CT2MR_Image: '../datasets/CT/CT2MR_image'
CT2MR_Label: '../datasets/CT/CT2MR_label'
```

## 2. Environment

- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## 3. Train style translation network

- Train stage one image-to-image translation model for style transfer

```bash
python stage_1_i2i_train.py --source_path '/mnt16t/FLARE24/Dataset/CT_2d' --target_path '/mnt16t/FLARE24/Dataset/MR_2d'
```

```bash
python stage_1.5_i2i_inference.py --ckpt_path YOUR_PATH --source_npy_dirpath SOURCE_PATH --target_npy_dirpath TARGET_PATH --save_nii_dirpath SAVE_PATH 
```


## References
* [DAR-UNet](https://github.com/Kaiseem/DAR-UNet)

## Acknowledgement

```bibtex
@article{yao2022darunet,
  title={A novel 3D unsupervised domain adaptation framework for cross-modality medical image segmentation},
  author={Yao, Kai and Su, Zixian and Huang, Kaizhu and Yang, Xi and Sun, Jie and Hussain, Amir and Coenen, Frans},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2022},
  publisher={IEEE}
}

@article{dorent2023crossmoda,
  title={CrossMoDA 2021 challenge: Benchmark of cross-modality domain adaptation techniques for vestibular schwannoma and cochlea segmentation},
  author={Dorent, Reuben and Kujawa, Aaron and Ivory, Marina and Bakas, Spyridon and Rieke, Nicola and Joutard, Samuel and Glocker, Ben and Cardoso, Jorge and Modat, Marc and Batmanghelich, Kayhan and others},
  journal={Medical Image Analysis},
  volume={83},
  pages={102628},
  year={2023},
  publisher={Elsevier}
}
```
