import os
import glob
import json  # 新增: 引入 json 模組
import numpy as np

def load_config(config_path):
    """
    讀取 JSON 設定檔
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到設定檔: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_file_ids(dataset_root, subset='train'):
    """
    掃描資料夾，取得該子集下所有的檔案 ID。
    """
    rgb_dir = os.path.join(dataset_root, subset, 'images', 'rgb')

    if not os.path.exists(rgb_dir):
        raise FileNotFoundError(f"找不到路徑: {rgb_dir}")

    files = glob.glob(os.path.join(rgb_dir, "*.jpg"))
    file_ids = [os.path.splitext(os.path.basename(f))[0] for f in files]

    return sorted(file_ids)

def make_output_dirs(output_root, subset):
    """
    建立輸出資料夾結構
    """
    img_out = os.path.join(output_root, subset, 'images')
    lbl_out = os.path.join(output_root, subset, 'labels')

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    return img_out, lbl_out

def save_npy(path, data):
    np.save(path, data)
