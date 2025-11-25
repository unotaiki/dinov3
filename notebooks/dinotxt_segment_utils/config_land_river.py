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
MODE = "whole" # "whole" or "slice"
RESIZE = 1024
LOGIT_SCALE = 30.0
SIDE = 384
STRIDE = 192
CHUNK_SIZE = 32

AGGREGATION = "sum" # "sum" or "max" or "mean"

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
    "river", "lake", "sea",
    "river water",      # 基本
    "blue water",       # 晴天時
    "green water",      # 深い川や藻類の影響
    "water surface",    # 表面
    "ripples",          # さざ波（テクスチャ特徴）
    "reflection",       # 水面反射
    "stream",           # 流れ
    "fluid",             # 液体
    "riverbeds",
]

# カテゴリ2: 陸地 (抽出対象1)
PROMPTS_LAND = [
# --- 陸地・砂州・岩 ---
    "sand",             # 砂
    "gravel",           # 砂利（砂州によくある）
    "pebbles",          # 小石
    "dirt",             # 土
    "dry land",         # 乾いた土地
    "rock", "stone",    # 岩、石
    "boulder",          # 大きな岩
    "concrete",         # 護岸など
    "shore",            # 岸辺
    "ground",           # 地面
    "reef",             # 砂州
    "sandbank",         # 砂州
    "sandy spit",       # 砂州
    "shoal",            # 砂州
    "shore",            # 岸辺

    "grass",            # 草
    "tree", "bush",     # 木、茂み
    "vegetation",
    "plant",
    "grass on the shore"
]

# # カテゴリ3: 植生 (抽出対象2)
# PROMPTS_VEGETATION = [
#     # --- 一般的な植生  ---
#     "vegetation",       # 植生一般
#     "grass",            # 草
#     "tree", "bush",     # 木、茂み
#     "leaves",           # 葉
#     "plant",
#     # --- 浮草  ---
#     "leaves on the water surface",
#     # "green scum",       # アオコ、浮遊物
#     # "floating weeds",   # 浮草
#     # "water plants",     # 水草
#     # "lily pads",        # 睡蓮の葉
#     # # --- 苔類  ---
#     # "algae",            # 藻（水面の緑を吸着させる）
#     # "moss",             # 苔
# ]

# マッピング定義: どのリストがどのクラスIDに対応するか
# 0: Water, 1: Land, 2: Vegetation
CLASS_MAPPING = {
    0: PROMPTS_WATER,
    1: PROMPTS_LAND
}

CLASS_NAMES = ["Water", "Land"]