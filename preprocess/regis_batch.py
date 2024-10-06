# 需要创建好一个delay文件夹(delay_root)，把所有delay模态放进去，其他模态放另一个文件夹(input_root)
import ants
import os
from tqdm import tqdm


def regis_data(input_root, delay_root, save_root):
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for image_name in tqdm(os.listdir(input_root)):
        save_path = os.path.join(save_root, image_name)
        move_path = os.path.join(input_root, image_name)

        if os.path.exists(save_path):
            print("have been run, skip")
        else:
            print("begin regis:", image_name)

            modality = image_name.split("_")[2]
            fix_name = image_name.replace(modality, "C+Delay")
            fix_path = os.path.join(delay_root, fix_name)
            try:
                fix_img = ants.image_read(fix_path)
                move_img = ants.image_read(move_path)
                outs = ants.registration(fix_img, move_img, type_of_transforme="Syn")
                reg_img = outs["warpedmovout"]
                ants.image_write(reg_img, save_path)
                print(f"finish: {image_name}")

            except Exception as e:
                print(f"find {image_name} err: {str(e)}")
                continue
