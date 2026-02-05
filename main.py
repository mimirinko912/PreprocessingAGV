import argparse
import os
from tqdm import tqdm
from src.io_utils import get_file_ids, make_output_dirs, save_npy, load_config
from src.img_utils import load_merged_image, apply_boundary
from src.label_utils import merge_labels

def main():
    parser = argparse.ArgumentParser(description="Agriculture-Vision 2021 Preprocessor")

    # 這裡將 required=True 拿掉，並加入 default=None，以便讓 Config 接手
    parser.add_argument("--config", type=str, default="config.json", help="設定檔路徑 (預設 config.json)")
    parser.add_argument("--input_dir", type=str, default=None, help="原始資料集根目錄 (會覆蓋 config)")
    parser.add_argument("--output_dir", type=str, default=None, help="輸出 NPY 目錄 (會覆蓋 config)")
    parser.add_argument("--subset", type=str, default=None, choices=["train", "val", "test"], help="子集 (會覆蓋 config)")

    args = parser.parse_args()

    # 1. 載入 Config
    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"❌ 無法讀取設定檔: {e}")
        return

    # 2. 參數優先權邏輯：Command Line Args > Config File
    input_dir = args.input_dir if args.input_dir else cfg.get('input_dir')
    output_dir = args.output_dir if args.output_dir else cfg.get('output_dir')
    subset = args.subset if args.subset else cfg.get('subset', 'train') # 預設 train

    # 檢查必要參數
    if not input_dir or not output_dir:
        print("❌ 錯誤: 必須設定 input_dir 和 output_dir (在 config.json 或指令參數中)")
        return

    print(f"🚀 開始處理 Agriculture-Vision 2021 [{subset}]")
    print(f"📄 讀取設定: {args.config}")
    print(f"📂 輸入: {input_dir}")
    print(f"📂 輸出: {output_dir}")

    # 3. 準備輸出資料夾
    img_out_dir, lbl_out_dir = make_output_dirs(output_dir, subset)

    # 4. 取得檔案列表
    try:
        file_ids = get_file_ids(input_dir, subset)
        print(f"📊 共發現 {len(file_ids)} 筆資料")
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        return

    # 5. 處理迴圈
    for file_id in tqdm(file_ids, desc="Processing"):
        try:
            # --- 處理影像 ---
            img = load_merged_image(input_dir, subset, file_id)
            img = apply_boundary(img, input_dir, subset, file_id)
            save_npy(os.path.join(img_out_dir, f"{file_id}.npy"), img)

            # --- 處理標籤 (Test set 除外) ---
            if subset != 'test':
                label = merge_labels(input_dir, subset, file_id, img.shape)
                save_npy(os.path.join(lbl_out_dir, f"{file_id}.npy"), label)

        except Exception as e:
            print(f"\n⚠️ 處理檔案 {file_id} 時發生錯誤: {e}")
            continue

    print("\n✅ 所有處理完成！")

if __name__ == "__main__":
    main()
