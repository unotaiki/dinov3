import math
import torch
import torch.nn.functional as F
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVTF
from torch import Tensor, nn
from PIL import Image
import numpy as np
import cv2

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