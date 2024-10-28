# Data Preparing

## Image-to-image translation datasets for stage one

1. Respacing the source and target domain images (both should be gray) with the same XY plane resolutions, and crop/pad to the size of [512, 512, d] in terms of [width, height, depth].

2. Normlizae each 3D images to [0, 1], and extract 2D slices from 3D volumes along depth-axis.

3. Stack the list of 2D slices at zero dimension for the two domains respectively, resulting in 3D tensor with size of [N, 512, 512], and then save them as the follows:

```bash
.
└── DARUNET
    └──datasets
            ├── A_imgs.npy
            └── B_imgs.npy
```
