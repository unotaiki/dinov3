import dataclasses
import math
import warnings
from typing import Callable
import os
import glob
import datetime
import numpy as np
import tqdm
from regex import template

from PIL import Image
import cv2
import lovely_tensors
import torch
import torch.nn.functional as F
from torch.amp import autocast
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVTF
from omegaconf import OmegaConf
from torch import Tensor, nn
from torchmetrics.classification import MulticlassJaccardIndex


DINOv3_REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # Please add here the path to your DINOv3 repository

# dino.txt (ヘッド部分) の重み
TEXT_HEAD_WEIGHTS = os.path.join(DINOv3_REPO_DIR, "checkpoints", "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth")
# ViT-L (バックボーン) の重み
BACKBONE_WEIGHTS = os.path.join(DINOv3_REPO_DIR, "checkpoints", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")

# 可視化
LOGIT_SCALE = 30.0

# 1. 画像が入っているフォルダ
INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "mask_data"))
INPUT_DIR_NAME = os.path.basename(INPUT_DIR)

# 2. マスク画像の保存先
NOW = datetime.datetime.now().strftime('%Y%m%d_%H%M')
OUTPUT_DIR = os.path.join(DINOv3_REPO_DIR, "notebooks", "output", INPUT_DIR_NAME, NOW)

# ディレクトリ作成
os.makedirs(os.path.join(OUTPUT_DIR, "mask"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "prob"), exist_ok=True)


# 3. 検出したい物体 (これが見つかったら白くなります)
TARGET_PROMPTS = [
    "sky", "cloud", "ground", "grass", "tree", "building", 
    "road", "dry land", "vegetetation", "nothing", "people", "tool", "drone" 
]

# 4. 背景や除外したいもの (これらは黒くなります)
# ※ 重要: モデルに「ターゲット以外」の選択肢を与えるため、想定される背景を記述してください
BACKGROUND_PROMPTS = [
    "water surface", "river", "lake", "sea", "water"  
]

# 5. 推論モード (精度優先なら slide, 速度優先なら whole)
INFERENCE_MODE = "slide" # 'slide' or 'whole'
RESIZE = 512             # short side of input images

# Only used for mode=slide
SIDE: int = 384
STRIDE: int = 192

DRONE_PROMPT_TEMPLATES = (
    "a photo of the {0}.",
    "a photo of a {0}.",
    "a satellite imagery of {0}.",
    "a aerial view of {0}.",
    "{} top view.",
    "an overhead view of {0}.",        # 真上からの視点
    "a top-down photo of {0}.",        # 真下を見下ろす視点
    "a drone view of a {0}.",          # ドローン視点であることを明示
    "a bird's-eye view of {0}.",       # 鳥瞰図
    "a high-angle shot of {0}.",       # 高角度からの撮影
    "a {0} viewed from above.",        # 上から見た物体
    "a {0} seen from the sky.",        # 空から見た物体
    "an aerial photography of {0}.",   # 航空写真
    "a low-altitude aerial shot of {0}.", # 低空飛行時の視点
    "a fisheye lens photo of {0}.",    # アクションカメラ等に多い魚眼
    "a wide-angle drone shot of {0}.", # 広角レンズ
    "a {0} on the ground.",            # 地面にあることを強調（背景分離）
    "a tiny {0} seen from far away.",  # 高高度で小さく見える場合
    "an orthophoto of {0}.",           # オルソ画像（地図用補正画像）
    "UAV imagery of {0}.",             # 無人航空機
    "FPV drone footage of {0}.",       # 高速・傾きのある視点
    "a {0} in a landscape.",           # 風景の中の物体
)

ALL_PROMPT_TEMPLATES =  DRONE_PROMPT_TEMPLATES

ALL_CLASS_NAMES = TARGET_PROMPTS + BACKGROUND_PROMPTS
NUM_TARGETS = len(TARGET_PROMPTS)
TARGET_INDICES = list(range(NUM_TARGETS))
BG_START_IDX = NUM_TARGETS

print(f"Targets: {TARGET_PROMPTS}")
print(f"Backgrounds: {BACKGROUND_PROMPTS}")

# ============
# Load model
# ============
import sys
sys.path.append(DINOv3_REPO_DIR)

print("Loading model...")
model, tokenizer = torch.hub.load(
    DINOv3_REPO_DIR, 
    'dinov3_vitl16_dinotxt_tet1280d20h24l', 
    source='local', 
    weights=TEXT_HEAD_WEIGHTS,
    backbone_weights=BACKBONE_WEIGHTS,
    verbose=True
)

model.to("cuda", non_blocking=True)
model.eval()
tokenizer = tokenizer.tokenize


# ============
# テキスト特徴量
# ============

text_feats = []   # 最終的に、すべてのクラス（犬、猫、etc.）の計算結果（特徴量）が順番に格納
CHUNK_SIZE = 32   # AIの計算はRAMを大量に使う。多数のデータを一度に処理するとメモリ不足でエラーになるため、小分けにして処理

print("Encoding text prompts...")
with torch.no_grad():    # 今は学習中じゃないから、余計な計算記録はしなくていいよ」と命令することで、メモリを節約し、計算速度を向上
    for class_name in tqdm.tqdm(ALL_CLASS_NAMES, desc="Classes"):
        all_prompts = [template.format(class_name) for template in ALL_PROMPT_TEMPLATES]
        class_embeddings = []

        # Chunk処理でメモリを節約
        for i in range(0, len(all_prompts), CHUNK_SIZE):
            batch_prompt = all_prompts[i : i + CHUNK_SIZE]
            tokens = tokenizer(batch_prompt).to("cuda", non_blocking=True) # 文章をバラバラにして、辞書にあるID番号（トークン）に変換。例：「a photo of dog」→ [49, 856, 11, 340]
            with autocast('cuda', enabled=True):                                   # 小数点の精度を自動調整して（例えば32ビットから16ビットへ）、計算を高速化・省メモリ化する機能
                feats = model.encode_text(tokens)                          # AIモデルに数字の列（トークン）を渡し、その文章の意味を表すベクトを生成
            feats = feats[:, feats.shape[1] // 2 :]                        # 元の特徴量のうち後半半分だけを抽出して、新しいfeatsとする操作です。これは、特定のニューラルネットワークアーキテクチャ（例えば、ある種の埋め込み手法や特徴の冗長性を減らす手法）で行われる
            feats = F.normalize(feats, p=2, dim=-1)                        # L2ノルムで、dim=-1つまり最後の次元（通常、個々の特徴ベクトルが格納されている次元）で正規化(長さを一で向きだけに)
            class_embeddings.append(feats)
            
        feats = torch.cat(class_embeddings, dim=0)                         # Chunk処理したfeatsを縦にガッチャンコ
        feats = feats.mean(dim=0)                                          # 平均プーリング: ここで、「写真の犬」「スケッチの犬」など数十種類のプロンプトの特徴量をすべて足して割
        feats = F.normalize(feats, p=2, dim=-1)                            # 平均をとると矢印の長さが変わってしまうので、再度長さを1に
        text_feats.append(feats)
    
text_feats = torch.stack(text_feats)                                       # リストに入っていた個別の特徴量（犬、猫、車…）を積み重ねて（Stack）、1つの巨大な行列（Tensor）に変換





# ============
# Text情報(EncodeされたVector)と画像からセマンティックセグメンテーション
# ============

# ============
# 処理に必要な関数
# ============

# 画像の縦横比を保ったまま、短い辺を指定サイズに縮小。巨大すぎる画像を小さくして計算を速くするため
class ShortSideResize(nn.Module):
    def __init__(self, size: int, interpolation: TVT.InterpolationMode) -> None:
        super().__init__()
        self.size = size
        self.interpolation = interpolation

    def forward(self, img: Tensor) -> Tensor:
        _, h, w = TVTF.get_dimensions(img)
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img
        if w < h:
            new_w = self.size
            new_h = int(self.size * h / w)
            return TVTF.resize(img, [new_h, new_w], self.interpolation)
        else:
            new_h = self.size
            new_w = int(self.size * w / h)
            return TVTF.resize(img, [new_h, new_w], self.interpolation)

# 学習対象の ImageNet の標準的な色味
NORMALIZE_IMAGENET = TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def encode_image(model, img: Tensor) -> tuple[Tensor, Tensor]:
    """Extract image features from the backbone and the additional blocks."""
    B, _, H, W = img.shape
    P = model.visual_model.backbone.patch_size # In the case of our DINOv3
    new_H = math.ceil(H / P) * P
    new_W = math.ceil(W / P) * P

    # Stretch image to a multiple of patch size
    if (H, W) != (new_H, new_W):
        img = F.interpolate(img, size=(new_H, new_W), mode="bicubic", align_corners=False)  # [B, 3, H', W']

    B, _, h_i, w_i = img.shape

    backbone_patches = None
    cls_tokens, _, patch_tokens = model.visual_model.get_class_and_patch_tokens(img)
    blocks_patches = (
        patch_tokens.reshape(B, h_i // P, w_i // P, -1).contiguous()
    ) # [1, h, w, D]

    return backbone_patches, blocks_patches

def predict_whole(model, img: Tensor, text_features: Tensor) -> Tensor:
    # Extract image features from the additional blocks, ignore the backbone features
    _, H, W = img.shape
    _, blocks_feats = encode_image(model, img.unsqueeze(0))  # [1, h, w, D]
    _, h, w, _ = blocks_feats.shape
    blocks_feats = blocks_feats.squeeze(0)  # [h, w, D]

    # Cosine similarity between patch features and text features (already normalized)
    blocks_feats = F.normalize(blocks_feats, p=2, dim=-1)  # [h, w, D]
    cos = torch.einsum("cd,hwd->chw", text_features, blocks_feats)  # [num_classes, h, w]

    # Return low-res cosine similarities, they will be upsampled to the target resolution later
    return cos

def predict_slide(model, img: Tensor, text_features: Tensor, side: int, stride: int) -> Tensor:
    # Iterate over overlapping windows, accumulate predictions at the image resolution
    _, H, W = img.shape  
    num_classes, _ = text_features.shape
    probs = torch.zeros([num_classes, H, W], device="cuda")
    counts = torch.zeros([H, W], device="cuda")
    h_grids = max(H - side + stride - 1, 0) // stride + 1
    w_grids = max(W - side + stride - 1, 0) // stride + 1
    for i in range(h_grids):
        for j in range(w_grids):
            y1 = i * stride
            x1 = j * stride
            y2 = min(y1 + side, H)
            x2 = min(x1 + side, W)
            y1 = max(y2 - side, 0)
            x1 = max(x2 - side, 0)

            # Compute cosine similarities for this window, same logic as predict_whole
            img_window = img[:, y1:y2, x1:x2]  # [3, H_win, W_win]
            cos = predict_whole(model, img_window, text_features)  # [num_classes, h, w]

            # Upsample to the window resolution and accumulate "probabilities"
            # NOTE: they aren't real probabilities, just the result of applying softmax to cosine similarities
            cos = F.interpolate(
                cos.unsqueeze(0),
                size=img_window.shape[1:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)  # [num_classes, H_win, W_win]
            probs[:, y1:y2, x1:x2] += cos.softmax(dim=0)  # [num_classes, h, w]
            counts[y1:y2, x1:x2] += 1
    probs /= counts

    # Return "probabilities" at the img resolution, they will be upsampled to the target resolution later
    return probs  # [num_classes, H, W]

# ============
# メインパート
# ============
image_paths = glob.glob(os.path.join(INPUT_DIR, "*"))       # 指定したフォルダの中にあるファイル名をすべて取ってくる
valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
image_paths = [p for p in image_paths if os.path.splitext(p)[-1].lower() in valid_extensions]

print(f"Found {len(image_paths)} images in {INPUT_DIR}")

# Transformer の定義
transform = TVT.Compose([
    ShortSideResize(RESIZE, TVT.InterpolationMode.BICUBIC),
    TVT.ToTensor(),                                            # データ形式: 画像データは通常「0〜255の整数」ですが、これを「0.0〜1.0の小数（Tensor形式）」に変換
    NORMALIZE_IMAGENET,                                        # 正規化: 画像の色味を、AIが学習に使った「ImageNet」という標準的なデータセットの平均的な色味に合わせる
])

model.eval()                                                   # AI Model を学習モードから評価モードに切り替え
with torch.no_grad():
    for image_path in tqdm.tqdm(image_paths, desc="Processing"):

        # Load a image
        original_image = Image.open(image_path).convert("RGB")          # 白黒画像や、透明度付き画像を統一形式に
        W_orig, H_orig = original_image.size

        # Preprocess
        img_tensor = transform(original_image).unsqueeze(0).to("cuda", non_blocking=True)   # .unsqueeze(0) : AIは画像を「1枚」ではなく、常に「束（バッチ）」で受け取ることを想定.  [色, 高さ, 幅] → [1, 色, 高さ, 幅]
        _, _, H, W = img_tensor.shape

        # Inference
        with autocast('cuda', enabled=True):                                   # 小数点の精度を自動調整して（例えば32ビットから16ビットへ）、計算を高速化・省メモリ化する機能
            if INFERENCE_MODE == "whole":
                output = predict_whole(model, img_tensor.squeeze(0), text_feats)
                need_softmax = True      # predict_whole は Cosine Similarity を返すと仮定 (Softmaxが必要)
            elif INFERENCE_MODE == "slide":
                output = predict_slide(model, img_tensor.squeeze(0), text_feats, SIDE, STRIDE)
                need_softmax = False     # predict_slide は既に Softmax済みの確率を返すと仮定

        # Temperature (Logit Scale)
        # コサイン類似度は -1.0 〜 1.0 の範囲.  この小さい値をそのまま softmax に通すと、どのクラスの確率も「どんぐりの背比べ」になり、確信度が平準化（例：どのクラスも確率0.05付近）
        output = output * LOGIT_SCALE

        # Post Process: resize to original size
        output = F.interpolate(output.unsqueeze(0), size=(H_orig, W_orig), mode="bilinear", align_corners=False)    # pred shape: [num_classes, H_feat, W_feat] -> interpolate -> [1, num_classes, H_orig, W_orig]
        # Batch次元を削除して [C, H, W] に戻す
        output = output.squeeze(0)

        # HEATMAP
        if need_softmax:
            probs = F.softmax(output, dim=0) # [C, H, W], dim=0はClass次元
        else:
            probs = output # すでに確率値


# ==========================================
        # 1. MASK 生成ロジック (修正版)
        # ==========================================
        # 各ピクセルで最も確率が高いクラスのインデックスを取得
        pred_mask = probs.argmax(dim=0).cpu().numpy() # [H, W]

        # 出力用マスク初期化 (黒)
        final_mask = np.zeros_like(pred_mask, dtype=np.uint8)

        # 「予測クラスID」が「ターゲットIDリスト」に含まれる場所を 255(白) にする
        # np.isin を使うと高速かつ簡潔です
        is_target = np.isin(pred_mask, TARGET_INDICES)
        final_mask[is_target] = 255


        # ==========================================
        # 2. HEATMAP 生成ロジック (修正版)
        # ==========================================
        # コンセプト: 
        #  - そのピクセルが「背景クラス」と判定された場合 → その確率値をヒートマップにする
        #  - そのピクセルが「ターゲットクラス」と判定された場合 → 黒にする

        # 最大確率値とそのインデックスを取得
        max_val, max_idx = torch.max(probs, dim=0) # max_val: [H, W], max_idx: [H, W]

        # 背景クラスかどうかを判定 (インデックスが BG_START_IDX 以上なら背景)
        is_bg_mask = max_idx >= BG_START_IDX  # Bool Tensor [H, W]

        # ヒートマップ用の生データ作成 (0.0 ~ 1.0)
        prob_map_tensor = torch.zeros_like(max_val)
        
        # 背景と判定された場所だけ、その確率値を入れる (それ以外は0.0のまま)
        prob_map_tensor[is_bg_mask] = max_val[is_bg_mask]

        # NumPy / uint8 変換
        prob_map_numpy = prob_map_tensor.cpu().numpy()
        heatmap_uint8 = (prob_map_numpy * 255).astype(np.uint8)

        # カラーマップ適用 (Viridis: 0=紫, 255=黄)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)

        # applyColorMapは値が0の場所(ターゲット部分)を「紫」にしてしまうため、
        # 背景以外(ターゲット部分)を強制的に「黒 (0,0,0)」で塗りつぶす
        bg_mask_numpy = is_bg_mask.cpu().numpy()
        heatmap_color[~bg_mask_numpy] = [0, 0, 0] # ~ はNOT演算子

        # 保存
        filename = os.path.basename(image_path) 
        name, _ = os.path.splitext(filename)
        mask_path = os.path.join(OUTPUT_DIR, "mask", f"{name}.png")
        prob_path = os.path.join(OUTPUT_DIR, "prob", f"{name}.png")

        cv2.imwrite(mask_path, final_mask)
        cv2.imwrite(prob_path, heatmap_color)

print(f"Done! Check the output directory{OUTPUT_DIR}.")
