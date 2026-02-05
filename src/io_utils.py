import os
import glob
import numpy as np

def get_file_ids(dataset_root, subset='train'):
    """
    掃描資料夾，取得該子集下所有的檔案 ID (不含副檔名)。
    以 images/rgb 資料夾內的檔案為基準。
    """
    rgb_dir = os.path.join(dataset_root, subset, 'images', 'rgb')

    if not os.path.exists(rgb_dir):
        raise FileNotFoundError(f"找不到路徑: {rgb_dir}")

    # 取得所有 .jpg 檔案
    files = glob.glob(os.path.join(rgb_dir, "*.jpg"))
    # 提取檔名 (去掉路徑和副檔名)
    file_ids = [os.path.splitext(os.path.basename(f))[0] for f in files]

    return sorted(file_ids)

def make_output_dirs(output_root, subset):
    """
    建立輸出資料夾結構：
    output_root/subset/images
    output_root/subset/labels
    """
    img_out = os.path.join(output_root, subset, 'images')
    lbl_out = os.path.join(output_root, subset, 'labels')

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    return img_out, lbl_out

def save_npy(path, data):
    """
    將 numpy array 儲存為 .npy 檔
    """
    np.save(path, data)
