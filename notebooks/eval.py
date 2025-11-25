import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd

# ==========================================
# 設定エリア
# ==========================================

# クラス定義 (推論時と同じHEXコードを使用してください)
CLASS_MAPPING = {
    0: "#658FFF",  # Water (例として指定された色)
    1: "#CC9933",  # Land (例: Forest Green)
    2: "#66FF66", # vegetation 
}

CLASS_NAMES = {
    0: "Water",
    1: "Land",
    2: "Vegetation"
}

# ==========================================
# ヘルパー関数
# ==========================================

def hex_to_bgr(hex_str):
    """HEX文字列 (#RRGGBB) を OpenCV 用の (B, G, R) リストに変換"""
    hex_str = hex_str.lstrip('#')
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return [b, g, r]

def color_img_to_mask(img, class_mapping):
    """
    色付き画像 (H, W, 3) を クラスIDマスク (H, W) に変換する。
    定義されていない色は -1 (無視) とする。
    """
    H, W, _ = img.shape
    # 初期値は -1 (Ignore/Background)
    mask = np.full((H, W), -1, dtype=np.int32)

    for class_id, hex_code in class_mapping.items():
        target_bgr = hex_to_bgr(hex_code)
        
        # OpenCVの画像データはnumpy配列なので、色の完全一致を探す
        # (注意: JPEG圧縮などのノイズがあると完全一致しないため、入力はPNG推奨)
        
        # 高速化のためのブロードキャスト比較
        # lower/upper boundを使って多少の色のズレを許容する場合は cv2.inRange を使うが
        # ここでは厳密な一致を想定
        matches = np.all(img == target_bgr, axis=-1)
        mask[matches] = class_id

    return mask

def compute_confusion_matrix(pred_mask, gt_mask, num_classes):
    """
    1枚の画像ペアから混同行列 (Confusion Matrix) を計算
    """
    # 無視するラベル (-1) を除外
    mask = (gt_mask >= 0) & (gt_mask < num_classes)
    
    label = num_classes * gt_mask[mask].astype(int) + pred_mask[mask].astype(int)
    count = np.bincount(label, minlength=num_classes**2)
    confusion_matrix = count.reshape(num_classes, num_classes)
    
    return confusion_matrix

# ==========================================
# メイン処理
# ==========================================

def main(pred_dir, gt_dir):
    print(f"Pred Dir: {pred_dir}")
    print(f"GT Dir  : {gt_dir}")

    pred_files = [f for f in os.listdir(pred_dir) if f.endswith(('.png', '.jpg', '.bmp'))]
    
    num_classes = len(CLASS_MAPPING)
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    # ファイル名マッチング処理
    matched_count = 0

    for filename in tqdm(pred_files, desc="Evaluating"):
        pred_path = os.path.join(pred_dir, filename)
        gt_path = os.path.join(gt_dir, filename)

        if not os.path.exists(gt_path):
            # 拡張子が違う場合の対応 (例: predはpng, gtはjpg)
            # 必要ならここで拡張子変換ロジックを入れる
            continue

        # 画像読み込み
        img_pred = cv2.imread(pred_path)
        img_gt = cv2.imread(gt_path)

        if img_pred is None or img_gt is None:
            print(f"Error reading {filename}")
            continue

        # サイズが異なる場合はリサイズ (GTに合わせる)
        if img_pred.shape != img_gt.shape:
            img_pred = cv2.resize(img_pred, (img_gt.shape[1], img_gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 色 -> クラスIDマスク変換
        mask_pred = color_img_to_mask(img_pred, CLASS_MAPPING)
        mask_gt = color_img_to_mask(img_gt, CLASS_MAPPING)

        # 混同行列の更新
        cm = compute_confusion_matrix(mask_pred, mask_gt, num_classes)
        total_cm += cm
        matched_count += 1

    if matched_count == 0:
        print("画像ペアが見つかりませんでした。ファイル名が一致しているか確認してください。")
        return

    # ==========================================
    # 指標計算
    # ==========================================
    
    # IoU = TP / (TP + FP + FN)
    intersection = np.diag(total_cm)
    union = total_cm.sum(axis=1) + total_cm.sum(axis=0) - intersection
    
    iou = intersection / (union + 1e-10) * 100
    pixel_acc = np.diag(total_cm).sum() / (total_cm.sum() + 1e-10) * 100
    
    # 結果表示用のデータフレーム作成
    results = []
    for i in range(num_classes):
        cls_name = CLASS_NAMES.get(i, str(i))
        results.append({
            "Class ID": i,
            "Class Name": cls_name,
            "IoU (%)": iou[i],
            "Total Pixels (GT)": total_cm[i, :].sum(),
            "Correct Pixels": intersection[i]
        })

    df = pd.DataFrame(results)
    mean_iou = np.nanmean(iou)

    print("\n" + "="*40)
    print(" Evaluation Report ")
    print("="*40)
    print(f"Processed Images: {matched_count}")
    print(f"Global Pixel Accuracy: {pixel_acc:.2f} %")
    print(f"Mean IoU (mIoU):       {mean_iou:.2f} %")
    print("-" * 40)
    print(df.to_string(index=False, float_format="%.2f"))
    print("="*40)

if __name__ == "__main__":
    # 引数解析
    parser = argparse.ArgumentParser(description="Segmentation Evaluation Script based on Color")
    parser.add_argument("--pred_dir", type=str, required=True, help="推論結果(色付き)が入ったフォルダ")
    parser.add_argument("--gt_dir", type=str, required=True, help="正解データ(色付き)が入ったフォルダ")
    
    args = parser.parse_args()
    
    main(args.pred_dir, args.gt_dir)