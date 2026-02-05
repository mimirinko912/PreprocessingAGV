import os
import cv2
import numpy as np

def load_merged_image(root_dir, subset, file_id):
    """
    讀取 RGB 和 NIR 並合併為 4 通道影像 (H, W, 4)。
    """
    # 建構路徑
    rgb_path = os.path.join(root_dir, subset, 'images', 'rgb', f"{file_id}.jpg")
    nir_path = os.path.join(root_dir, subset, 'images', 'nir', f"{file_id}.jpg")

    # 讀取 RGB (OpenCV 預設是 BGR，需轉 RGB)
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"Missing RGB file: {rgb_path}")
    img_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 讀取 NIR (單通道)
    if not os.path.exists(nir_path):
        raise FileNotFoundError(f"Missing NIR file: {nir_path}")
    img_nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)

    # 確保尺寸一致 (防呆機制)
    if img_rgb.shape[:2] != img_nir.shape:
        img_nir = cv2.resize(img_nir, (img_rgb.shape[1], img_rgb.shape[0]))

    # 合併通道 (H, W, 3) + (H, W) -> (H, W, 4)
    img_4ch = np.dstack((img_rgb, img_nir))

    return img_4ch

def apply_boundary(img_4ch, root_dir, subset, file_id):
    """
    讀取邊界遮罩並將無效區域設為 0。
    """
    boundary_path = os.path.join(root_dir, subset, 'boundaries', f"{file_id}.png")

    if os.path.exists(boundary_path):
        # 讀取邊界 (0:無效, 255:有效)
        boundary = cv2.imread(boundary_path, cv2.IMREAD_GRAYSCALE)

        # 確保尺寸一致
        if img_4ch.shape[:2] != boundary.shape:
            boundary = cv2.resize(boundary, (img_4ch.shape[1], img_4ch.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 建立 Mask (布林值)
        mask = (boundary == 255)

        # 擴展維度以符合 4 通道: (H, W) -> (H, W, 4)
        mask_expanded = np.repeat(mask[:, :, np.newaxis], 4, axis=2)

        # 套用遮罩 (無效區域變黑)
        img_masked = img_4ch * mask_expanded.astype(np.uint8)

        return img_masked
    else:
        # 如果沒有 boundary 檔案，直接回傳原圖
        return img_4ch
