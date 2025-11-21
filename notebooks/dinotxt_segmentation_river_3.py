import argparse
import os
import glob
import datetime
import json
import tqdm
import torch
import torch.nn.functional as F
from torch.amp import autocast
from PIL import Image
import numpy as np

import dinotxt_segment_utils.config as config
import dinotxt_segment_utils.utils as utils


def get_args():
    parser = argparse.ArgumentParser(description="DINOv3 Segmentation Inference")
    
    # パス関連
    parser.add_argument("--input_dir", type=str, default=config.INPUT_DIR_DEFAULT, help="Input image directory")
    parser.add_argument("--output_root", type=str, default=config.OUTPUT_ROOT_DEFAULT, help="Root directory for outputs")
    parser.add_argument("--repo_dir", type=str, default=config.DINOV3_REPO_DIR_DEFAULT, help="DINOv3 repository path")
    parser.add_argument("--head_weights", type=str, default=config.TEXT_HEAD_WEIGHTS_DEFAULT)
    parser.add_argument("--backbone_weights", type=str, default=config.BACKBONE_WEIGHTS_DEFAULT)

    # 推論設定
    parser.add_argument("--mode", type=str, default="slide", choices=["whole", "slide"], help="Inference mode")
    parser.add_argument("--resize", type=int, default=config.RESIZE, help="Short side resize size")
    parser.add_argument("--logit_scale", type=float, default=config.LOGIT_SCALE)
    
    return parser.parse_args()

def build_text_embeddings(model, tokenizer, device="cuda"):
    """
    全ての詳細プロンプトに対して埋め込みベクトルを生成し、
    [Total_Prompts, Dim] のテンソルと、
    各プロンプトがどのカテゴリ(Water/Land/Veg)に属するかのインデックスリストを返す
    """
    print("Encoding text prompts...")
    
    all_embeddings = []
    prompt_to_category_id = [] # 各プロンプトがどのカテゴリID(0,1,2)に対応するか

    # Configで定義された3カテゴリ(Water, Land, Veg)を順番に処理
    for category_id, prompts in config.CLASS_MAPPING.items():
        for p_text in prompts:
            # 1つの単語に対して複数のテンプレートを適用して平均を取る
            templates = [t.format(p_text) for t in config.PROMPT_TEMPLATES]
            
            with torch.no_grad(), autocast(device_type=device, enabled=True):
                tokens = tokenizer(templates).to(device, non_blocking=True)
                feats = model.encode_text(tokens)
                
                # 後半の特徴量を使用 & 正規化
                feats = feats[:, feats.shape[1] // 2 :]
                feats = F.normalize(feats, p=2, dim=-1)
                
                # テンプレート方向で平均 -> 再正規化 -> 1つのプロンプトに対するベクトル完成
                feat = feats.mean(dim=0)
                feat = F.normalize(feat, p=2, dim=-1)
                
                all_embeddings.append(feat)
                prompt_to_category_id.append(category_id)

    text_feats = torch.stack(all_embeddings) # [N, Dim]
    category_indices = torch.tensor(prompt_to_category_id, device=device) # [N]
    
    return text_feats, category_indices

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 出力ディレクトリ設定
    input_dir_name = os.path.basename(os.path.normpath(args.input_dir))
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = os.path.join(args.output_root, input_dir_name, timestamp)
    
    os.makedirs(os.path.join(output_dir, "mask"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "prob"), exist_ok=True)

    # 設定をJSON保存
    config_dict = vars(args)
    config_dict["prompts"] = config.CLASS_MAPPING # プロンプト情報も含める
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
    
    print(f"Output Directory: {output_dir}")

    # モデルロード
    model, tokenizer = utils.load_dinov3_model(
        args.repo_dir, args.head_weights, args.backbone_weights, device
    )
    tokenizer = tokenizer.tokenize

    # テキスト特徴量計算
    # text_feats: [全詳細プロンプト数, Dim]
    # category_map: [全詳細プロンプト数] (値は0=Water, 1=Land, 2=Veg)
    text_feats, category_map = build_text_embeddings(model, tokenizer, device)

    # 画像リスト取得
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = [p for p in glob.glob(os.path.join(args.input_dir, "*")) 
                   if os.path.splitext(p)[-1].lower() in valid_extensions]
    
    print(f"Found {len(image_paths)} images.")
    
    transform = utils.get_transform(args.resize)

    # 推論ループ
    with torch.no_grad():
        for image_path in tqdm.tqdm(image_paths, desc="Processing"):
            # 画像ロード
            original_image = Image.open(image_path).convert("RGB")
            W_orig, H_orig = original_image.size
            
            img_tensor = transform(original_image).unsqueeze(0).to(device, non_blocking=True)
            _, _, H, W = img_tensor.shape

            # 推論実行
            with autocast(device_type=device, enabled=True):
                if args.mode == "whole":
                    output = utils.predict_whole(model, img_tensor.squeeze(0), text_feats, config.LOGIT_SCALE)
                    probs_detailed = F.softmax(output, dim=0) # [詳細クラス数, H, W]
                else:
                    output = utils.predict_slide(model, img_tensor.squeeze(0), text_feats, config.SIDE, config.STRIDE, config.LOGIT_SCALE)
                    probs_detailed = output # [詳細クラス数, H, W]

            # リサイズして元の解像度に戻す
            probs_detailed = F.interpolate(
                probs_detailed.unsqueeze(0), 
                size=(H_orig, W_orig), 
                mode="bilinear", 
                align_corners=False
            ).squeeze(0)

            # ============================================================
            # Aggregation Logic (詳細プロンプトの確率を3カテゴリに集約)
            # ============================================================
            # Water, Land, Veg それぞれに属するチャネルの最大値(または合計)を取る
            # 「平均」するとベクトルが鈍るので、画素ごとに「最も反応した具体的な物体」のスコアを採用する (Max Aggregation)
            
            final_probs = []
            for cat_id in range(3): # 0, 1, 2
                # このカテゴリに属するインデックスを取得
                indices = (category_map == cat_id).nonzero(as_tuple=True)[0]
                if len(indices) > 0:
                    # そのカテゴリ内の詳細クラスの中で、最も高い確率を採用
                    cat_prob, _ = probs_detailed[indices].max(dim=0)
                    final_probs.append(cat_prob)
                else:
                    final_probs.append(torch.zeros((H_orig, W_orig), device=device))
            
            # [3, H, W] の確率マップ (0:Water, 1:Land, 2:Veg)
            three_class_probs = torch.stack(final_probs) 
            
            # 最終クラス決定 (0, 1, 2 のどれか)
            pred_mask = three_class_probs.argmax(dim=0).cpu().numpy() # [H, W]
            
            # 確信度マップ (表示用、最も高かったカテゴリの確率値)
            confidence_map = three_class_probs.max(dim=0)[0].cpu().numpy()

            # ============================================================
            # 保存
            # ============================================================
            filename = os.path.basename(image_path)
            name, _ = os.path.splitext(filename)
            
            # マスク保存: Water(0)は黒、Land(1)/Veg(2)は白
            utils.save_mask(pred_mask, os.path.join(output_dir, "mask", f"{name}.png"))
            
            # ヒートマップ保存: 背景(Water)以外を可視化
            utils.save_heatmap(confidence_map, pred_mask, os.path.join(output_dir, "prob", f"{name}.png"))

    print(f"Done! Check {output_dir}")

if __name__ == "__main__":
    main()