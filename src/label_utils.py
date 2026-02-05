import os
import cv2
import numpy as np

# 定義類別順序 (0 是背景)
CLASSES = [
    'background',            # 0
    'double_plant',          # 1
    'drydown',               # 2
    'endrow',                # 3
    'nutrient_deficiency',   # 4
    'planter_skip',          # 5
    'water',                 # 6
    'waterway',              # 7
    'weed_cluster'           # 8
]

def merge_labels(root_dir, subset, file_id, shape):
    """
    讀取所有類別資料夾，合併成一張單通道 Mask (H, W)。
    數值 0-8 對應上面的 CLASSES。
    """
    h, w = shape[:2]
    final_mask = np.zeros((h, w), dtype=np.uint8) # 初始化全黑 (背景=0)

    labels_root = os.path.join(root_dir, subset, 'labels')

    # 從索引 1 開始遍歷 (跳過 background)
    for class_id, class_name in enumerate(CLASSES[1:], start=1):
        class_path = os.path.join(labels_root, class_name, f"{file_id}.png")

        if os.path.exists(class_path):
            label_img = cv2.imread(class_path, cv2.IMREAD_GRAYSCALE)

            # 確保尺寸一致
            if label_img.shape != (h, w):
                label_img = cv2.resize(label_img, (w, h), interpolation=cv2.INTER_NEAREST)

            # 將該類別的位置填入 final_mask
            # 注意：這裡採用「後蓋前」策略，如果同一個像素有多個標籤，
            # index 較大的類別會覆蓋較小的。
            final_mask[label_img == 255] = class_id

    return final_mask
