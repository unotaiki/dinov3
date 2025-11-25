#!/bin/bash

# ==========================================
# 設定
# ==========================================
# Pythonスクリプトのパス
PYTHON_SCRIPT="notebooks/dinotxt_segmentation_river_3.py"

# 入力画像のルートディレクトリ
SOURCE_ROOT="/home/uno/projects/segment-drone/source"

# 処理対象のサブディレクトリリスト（画像に基づく）
TARGET_DIRS=(
    # "../data/mask_data"
    # "1_01_sasu"
    # "1_02_wo-RTK"
    # "1_03"
    "1_04"
    # "1_05_oblique"
    "1_06_manual"
)

# ==========================================
# 実行ループ
# ==========================================

for DIR_NAME in "${TARGET_DIRS[@]}"; do
    INPUT_PATH="${SOURCE_ROOT}/${DIR_NAME}"
    
    # ディレクトリが存在するか確認
    if [ -d "$INPUT_PATH" ]; then
        echo "----------------------------------------------------------------"
        echo "Starting inference for: ${DIR_NAME}"
        echo "Input Path: ${INPUT_PATH}"
        echo "Date: $(date)"
        echo "----------------------------------------------------------------"
        
        # Pythonスクリプト実行
        # 必要であれば --output_root や --mode などの引数をここに追加してください
        python3 "$PYTHON_SCRIPT" \
            --input_dir "$INPUT_PATH" \
        
        # 終了コードのチェック（エラーがあれば表示）
        if [ $? -eq 0 ]; then
            echo "Success: ${DIR_NAME}"
        else
            echo "Error occurred processing: ${DIR_NAME}"
        fi
        
        echo ""
    else
        echo "Warning: Directory not found -> ${INPUT_PATH}"
    fi
done

echo "All batch processes finished."