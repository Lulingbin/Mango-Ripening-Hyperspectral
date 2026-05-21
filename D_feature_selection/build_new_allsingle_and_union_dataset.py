import numpy as np
import pandas as pd
from pathlib import Path

# ---------- 用户配置 ----------
OUT_DIR  = Path(r"F:\study\manggo_program\mango_predict_all\pipeline\A_Dataset\FX10\Hardness_10_140\outlier_isolationforest\Dataset")
TOP_K    = 10
METHODS  = [ 'PCA4FS','RF4FS','SLP4FS','GA4FS']

# --------------------------------

def reduce(df, idx, target_Y):


    """按索引裁剪光谱列并保留目标变量和Polygon_ID"""
    band_cols = df.filter(regex="^Band_|^x\\d").columns
    reduced_cols = band_cols[idx]
    # 确保保留Polygon_ID列
    if 'Polygon_ID' in df.columns:
        return pd.concat([df[['Polygon_ID', target_Y]], df[reduced_cols]], axis=1)
    else:
        return pd.concat([df[[target_Y]], df[reduced_cols]], axis=1)

def build_fs(target_X,target_Y):

    fold_dir = OUT_DIR / f"{target_X}_{target_Y}_feature_selection_results"
    res_dir  = fold_dir

    # ---------- 1) 生成并保存每种方法的 10 波段子集 ----------
    for m in METHODS:
        idx = np.load(res_dir / f"{m}_top{TOP_K}.npy")  # 10 个索引
        for split in ["train", "val"]:
            df      = pd.read_csv(OUT_DIR / f"{split}.csv")
            new_df  = reduce(df, idx,target_Y)
            out_csv = fold_dir / f"{split}_{m}.csv"
            new_df.to_csv(out_csv, index=False)
            print(f"Fold {target_X}_{target_Y}_feature_selection_results | {split}_{m}.csv saved ({len(idx)} features).")

    # ---------- 2) 原有的并集逻辑 ----------
    union_idx = set()
    for m in METHODS:
        idx = np.load(res_dir / f"{m}_top{TOP_K}.npy")
        union_idx.update(idx)
    union_idx = np.sort(list(union_idx))

    for split in ["train", "val"]:
        csv_in  = OUT_DIR / f"{split}.csv"
        csv_out = fold_dir / f"{split}_union.csv"
        df = pd.read_csv(csv_in)
        new_df = reduce(df, union_idx,target_Y)
        new_df.to_csv(csv_out, index=False)
        print(f"Fold {target_X}_{target_Y} | {split}_union.csv saved ({len(union_idx)} features).")

    np.save(fold_dir / "union_features_idx.npy", union_idx)  # 保存并集特征索引

    # 读取原始数据集的列名
    train_df = pd.read_csv(OUT_DIR / "train.csv")
    band_cols = train_df.filter(regex="^Band_|^x\\d").columns
    wavelength_nm = [int(band_cols[i].split('_')[-1]) for i in union_idx]

    pd.DataFrame({
        "wavelength_index": union_idx,
        "wavelength_nm": wavelength_nm
    }).to_csv(fold_dir / "union_features.csv", index=False)  # 保存并集特征波长
    print("✅ All fold-wise union & per-method datasets created.")

    #原光谱数据也复制过来
    for split in ["train", "val"]:
        df = pd.read_csv(OUT_DIR / f"{split}.csv")
        """按索引裁剪光谱列并保留目标变量和Polygon_ID"""
        band_cols = df.filter(regex="^Band_|^x\\d").columns
        reduced_cols = band_cols
        # 确保保留Polygon_ID列
        if 'Polygon_ID' in df.columns:
            new_df = pd.concat([df[['Polygon_ID', target_Y]], df[reduced_cols]], axis=1)
        else:
            new_df = pd.concat([df[[target_Y]], df[reduced_cols]], axis=1)

        out_csv = fold_dir / f"{split}_all.csv"
        new_df.to_csv(out_csv, index=False)

        print(f"Fold {target_X}_{target_Y} | {split}_all.csv).")


ALL_Y = ["Hardness","TSS","carotenoids","starch_content"]
# ALL_X = ["FX10_and_FX17"]
# ALL_X = ["FX10"]
ALL_X = ["FX17"]


for xi in ALL_X:
    for yj in ALL_Y:
        build_fs(xi,yj)



