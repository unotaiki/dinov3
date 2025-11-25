import math
import torch
import torch.nn.functional as F
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVTF
from torch import Tensor, nn
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



# 学習対象の ImageNet の標準的な色味
NORMALIZE_IMAGENET = TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class ShortSideResize(nn.Module):
    """短い辺を指定サイズに合わせてリサイズする変換"""
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

def get_transform(resize_size: int):
    return TVT.Compose([
        ShortSideResize(resize_size, TVT.InterpolationMode.BICUBIC),
        TVT.ToTensor(),
        NORMALIZE_IMAGENET,
    ])

def load_dinov3_model(repo_dir, head_weights, backbone_weights, device="cuda"):
    """DINOv3モデルをロードする"""
    import sys
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)
    
    print(f"Loading model from {repo_dir}...")
    model, tokenizer = torch.hub.load(
        repo_dir,
        'dinov3_vitl16_dinotxt_tet1280d20h24l',
        source='local',
        weights=head_weights,
        backbone_weights=backbone_weights,
        verbose=True
    )
    model.to(device, non_blocking=True)
    model.eval()
    return model, tokenizer

def encode_image(model, img: Tensor) -> tuple[Tensor, Tensor]:
    """DINOv3バックボーンから画像特徴を抽出"""
    B, _, H, W = img.shape
    P = model.visual_model.backbone.patch_size
    new_H = math.ceil(H / P) * P
    new_W = math.ceil(W / P) * P

    if (H, W) != (new_H, new_W):
        img = F.interpolate(img, size=(new_H, new_W), mode="bicubic", align_corners=False)

    B, _, h_i, w_i = img.shape
    _, _, patch_tokens = model.visual_model.get_class_and_patch_tokens(img)
    blocks_patches = patch_tokens.reshape(B, h_i // P, w_i // P, -1).contiguous()
    return None, blocks_patches

def predict_slide(model, img: Tensor, text_features: Tensor, side: int, stride: int, logit_scale: float = 30.0) -> Tensor:
    _, H, W = img.shape
    num_classes, _ = text_features.shape
    probs = torch.zeros([num_classes, H, W], device=img.device)
    counts = torch.zeros([H, W], device=img.device)
    
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

            img_window = img[:, y1:y2, x1:x2]
            
            # 特徴量抽出 & コサイン類似度計算
            _, blocks_feats = encode_image(model, img_window.unsqueeze(0))
            blocks_feats = blocks_feats.squeeze(0)
            blocks_feats = F.normalize(blocks_feats, p=2, dim=-1)
            cos = torch.einsum("cd,hwd->chw", text_features, blocks_feats)
            
            # 補間 (Interpolate)
            cos = F.interpolate(
                cos.unsqueeze(0),
                size=img_window.shape[1:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            
            # ★★★ 修正ポイント: Softmaxの前に Scale を掛ける ★★★
            cos = cos * logit_scale
            
            # 確率に変換して加算
            probs[:, y1:y2, x1:x2] += cos.softmax(dim=0)
            counts[y1:y2, x1:x2] += 1
            
    probs /= counts
    return probs

def predict_whole(model, img: Tensor, text_features: Tensor, logit_scale: float = 30.0) -> Tensor:
    """画像全体を一括推論"""
    _, blocks_feats = encode_image(model, img.unsqueeze(0))
    blocks_feats = blocks_feats.squeeze(0)
    blocks_feats = F.normalize(blocks_feats, p=2, dim=-1)
    cos = torch.einsum("cd,hwd->chw", text_features, blocks_feats)
    cos = cos * logit_scale
    return cos # returns logits (needs softmax later)

def save_heatmap(prob_map_numpy, mask_numpy, save_path):
    """確率マップをヒートマップとして保存（背景は黒塗り）"""
    heatmap_uint8 = (prob_map_numpy * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
    
    # マスクが0（背景/Water）の部分は黒にする
    # mask_numpy: 1=Land/Veg, 0=Water と仮定
    heatmap_color[mask_numpy == 0] = [0, 0, 0]
    
    cv2.imwrite(save_path, heatmap_color)

def save_mask(mask_numpy, save_path):
    """2値マスク保存 (0:Black, 255:White)"""
    final_mask = np.zeros_like(mask_numpy, dtype=np.uint8)
    final_mask[mask_numpy > 0] = 255 # Water(0)以外を白にする
    cv2.imwrite(save_path, final_mask)



def save_semantic_segmentation_debug(
    similarity_map, 
    class_names, 
    save_path="debug_segmentation.png", 
    original_image=None,
    figsize=(15, 10)
):
    """
    各ピクセルがどのクラス（プロンプト）に分類されたかを色分けして保存するデバッグ関数。
    
    Args:
        similarity_map (numpy.ndarray or torch.Tensor): 
            形状が (H, W, C) または (C, H, W) の類似度スコア。
            Cはクラス数（プロンプトの総数）。
        class_names (list of str): 
            類似度マップのチャンネル順に対応するクラス名（プロンプト名）のリスト。
        save_path (str): 
            保存先のファイルパス。
        original_image (numpy.ndarray, optional): 
            比較用に元の画像を表示したい場合に渡す (H, W, 3)。Noneの場合はセグメンテーションのみ表示。
        figsize (tuple):
            画像のサイズ。
    """
    
    # 1. データ型の調整 (Tensor -> Numpy)
    if isinstance(similarity_map, torch.Tensor):
        similarity_map = similarity_map.detach().cpu().numpy()
    if original_image is not None and isinstance(original_image, torch.Tensor):
        original_image = original_image.detach().cpu().numpy()
        # (C, H, W) -> (H, W, C) もし必要なら
        if original_image.shape[0] == 3:
            original_image = np.transpose(original_image, (1, 2, 0))

    # 2. 形状の確認と修正 (C, H, W) -> (H, W, C)
    # チャンネル数がクラス名リストの長さと一致する次元を探す
    if similarity_map.shape[0] == len(class_names):
        similarity_map = np.transpose(similarity_map, (1, 2, 0))
    elif similarity_map.shape[-1] != len(class_names):
        raise ValueError(f"Map shape {similarity_map.shape} does not match class names length {len(class_names)}")

    # 3. Argmaxで最もスコアが高いクラスインデックスを取得
    # shape: (H, W)
    prediction_indices = np.argmax(similarity_map, axis=-1)

    # 4. プロットの作成
    if original_image is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax_img = axes[0]
        ax_seg = axes[1]
    else:
        fig, ax_seg = plt.subplots(1, 1, figsize=figsize)
        ax_img = None

    # 5. カラーマップの生成 (クラス数分だけ色を用意)
    # tab20は視認性の高い20色。クラス数がそれ以上の場合はgist_ncarなどを使用
    num_classes = len(class_names)
    cmap_name = 'tab20' if num_classes <= 20 else 'gist_ncar'
    cmap = plt.get_cmap(cmap_name, num_classes)

    # 6. セグメンテーションマップの描画
    im = ax_seg.imshow(prediction_indices, cmap=cmap, vmin=0, vmax=num_classes-1, interpolation='nearest')
    ax_seg.set_title("Semantic Segmentation Prediction")
    ax_seg.axis('off')

    # 7. 凡例（Legend）の作成
    # 各色がどのプロンプトに対応するかを表示
    values = np.unique(prediction_indices)
    colors = [im.cmap(im.norm(value)) for value in values]
    # 画像内に存在するクラスのみを凡例に表示する（見やすくするため）
    patches = [mpatches.Patch(color=colors[i], label=f"{class_names[values[i]]}") for i in range(len(values))]
    
    # 凡例を枠外に配置
    ax_seg.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # 8. 元画像の描画（ある場合）
    if ax_img is not None:
        # 正規化されている場合を考慮し、0-1または0-255に収める簡易処理
        if original_image.max() > 1.0:
            original_image = original_image.astype(np.uint8)
        else:
            original_image = np.clip(original_image, 0, 1)
            
        ax_img.imshow(original_image)
        ax_img.set_title("Original Image")
        ax_img.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Debug segmentation saved to: {save_path}")

def hex_to_bgr(hex_str):
    """HEX文字列 (#RRGGBB) を OpenCV 用の (B, G, R) タプルに変換"""
    hex_str = hex_str.lstrip('#')
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return (b, g, r)

def save_colored_segmentation(pred_mask, save_path, class_colors):
    """
    予測マスク(H, W)を受け取り、指定された色で塗り分けた画像を保存する
    
    Args:
        pred_mask (np.ndarray): [H, W] の形状を持つクラスインデックス (0, 1, ...) の配列
        save_path (str): 保存先のパス
        class_colors (dict): {class_id: "#HEXCODE"} の辞書
    """
    H, W = pred_mask.shape
    # カラー画像のキャンバスを作成 (H, W, 3)
    color_map = np.zeros((H, W, 3), dtype=np.uint8)

    # 定義された各クラスについて色を塗る
    for class_id, hex_code in class_colors.items():
        # 現在のクラスIDに該当するピクセルのマスクを作成
        mask = (pred_mask == class_id)
        
        # HEXをBGRに変換
        color_bgr = hex_to_bgr(hex_code)
        
        # 該当箇所を塗りつぶす
        color_map[mask] = color_bgr

    # 画像保存
    cv2.imwrite(save_path, color_map)