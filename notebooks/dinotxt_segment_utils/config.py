import os

# =========================
# パス設定 (デフォルト値)
# =========================
# 現在のファイルからの相対パスでリポジトリルートを推定
current_dir = os.path.dirname(os.path.abspath(__file__))
DINOV3_REPO_DIR_DEFAULT = os.path.abspath(os.path.join(current_dir, "..", ".."))

# 重みファイルのパス
TEXT_HEAD_WEIGHTS_DEFAULT = os.path.join(DINOV3_REPO_DIR_DEFAULT, "checkpoints", "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth")
BACKBONE_WEIGHTS_DEFAULT = os.path.join(DINOV3_REPO_DIR_DEFAULT, "checkpoints", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")

# 入出力
INPUT_DIR_DEFAULT = os.path.abspath(os.path.join(DINOV3_REPO_DIR_DEFAULT, "..", "data", "mask_data"))
OUTPUT_ROOT_DEFAULT = os.path.join(DINOV3_REPO_DIR_DEFAULT, "notebooks", "output")

# =========================
# 推論パラメータ
# =========================
RESIZE = 512
LOGIT_SCALE = 30.0
SIDE = 384
STRIDE = 192
CHUNK_SIZE = 32

# =========================
# プロンプト定義 (3クラス分類用)
# =========================

# テンプレート: ドローン視点・俯瞰視点を強調
PROMPT_TEMPLATES = [
    "a drone view of {}.",
    "an aerial view of {}.",
    "a top-down photo of {}.",
    "a photo of the {}.",
    "{} texture.",
    "a close-up of {}.",
    "a view of {} from above.",
    "an overhead view of {}.",
    "a bird's-eye view of {}.",
    "a high-angle shot of {}.",
    "an orthophoto of {}.",
]

# カテゴリ1: 水 (除外対象/黒マスク)
PROMPTS_WATER = [
    "river", "lake", "sea", "river water", "muddy water",
    "blue water", "green water", "water surface", "ripples",
    "reflection", "stream", "fluid", "water"
]

# カテゴリ2: 陸地 (抽出対象1)
PROMPTS_LAND = [
    "sand", "gravel", "pebbles", "dirt", "mud",
    "dry land", "rock", "stone", "boulder",
    "concrete", "shore", "ground", "road", "pavement",
    "soil"
]

# カテゴリ3: 植生 (抽出対象2)
PROMPTS_VEGETATION = [
    "algae", "moss", "green scum", "floating weeds",
    "water plants", "lily pads", "vegetation", "grass",
    "tree", "bush", "leaves", "forest", "plants"
]

# マッピング定義: どのリストがどのクラスIDに対応するか
# 0: Water, 1: Land, 2: Vegetation
CLASS_MAPPING = {
    0: PROMPTS_WATER,
    1: PROMPTS_LAND,
    2: PROMPTS_VEGETATION
}

CLASS_NAMES = ["Water", "Land", "Vegetation"]