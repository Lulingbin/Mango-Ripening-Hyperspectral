# preprocess_pipeline.py
import os, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from scipy.signal import savgol_filter
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# ---------- 用户配置 ----------
RAW_CSV = Path(r"F:\study\manggo_program\mango_predict_all\pipeline\A_Dataset\FX10\Hardness_10_140\outlier_isolationforest\clean.csv")
OUT_DIR = Path(r"F:\study\manggo_program\mango_predict_all\pipeline\A_Dataset\FX10\Hardness_10_140\outlier_isolationforest\Dataset")
WINDOW_LENGTH = 5
POLYORDER = 3
RANDOM_STATE = 42
# --------------------------------

OUT_DIR.mkdir(exist_ok=True, parents=True)

# ---------- 工具函数 ----------
def load_spectra(df: pd.DataFrame) -> np.ndarray:
    cols = df.filter(regex="^Band_|^x\\d").columns
    return df[cols].values.astype(np.float32)

def sg_filter(X: np.ndarray) -> np.ndarray:
    return savgol_filter(X, window_length=WINDOW_LENGTH, polyorder=POLYORDER, axis=1)

def snv_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=1, keepdims=True)
    sigma = X.std(axis=1, keepdims=True, ddof=0)
    return mu, sigma

def snv_transform(X, mu, sigma):
    return (X - mu) / (sigma + 1e-8)

def zscore_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X.mean(axis=0), X.std(axis=0, ddof=0)

def zscore_transform(X, mean, std):
    return (X - mean) / (std + 1e-8)

def save_csv(df: pd.DataFrame, spectra: np.ndarray, hardness: np.ndarray, save_path: Path):
    band_cols = df.filter(regex="^Band_|^x\\d").columns
    polygon_id = df["Polygon_ID"].values  # 获取 Polygon_ID 列

    TSS = df["TSS"].values  # 获取 Polygon_ID 列
    carotenoids = df["carotenoids"].values  # 获取 Polygon_ID 列
    starch_content = df["starch content"].values  # 获取 Polygon_ID 列

    pd.concat([pd.DataFrame({"Polygon_ID": polygon_id, "Hardness": hardness,"TSS": TSS,"carotenoids": carotenoids,"starch_content": starch_content},),  # 添加 Polygon_ID 列
               pd.DataFrame(spectra, columns=band_cols)], axis=1
              ).to_csv(save_path, index=False)


# ---------- 主流程 ----------
print("Loading single dataset...")
df = pd.read_csv(RAW_CSV)
X_raw = load_spectra(df)
y = df["Hardness"].values.astype(np.float32)
band_cols = df.filter(regex="^Band_|^x\\d").columns.tolist()

# 将目标变量 y 分为5个分位数区间
y_binned = pd.qcut(y, q=5, labels=False)

# 划分数据集（80%训练集，20%验证集），并保持目标变量 y 的分布
train_idx, val_idx = train_test_split(np.arange(len(X_raw)), test_size=0.2, random_state=RANDOM_STATE, stratify=y_binned)

# 创建输出目录
fold_dir = OUT_DIR / "fold_1"
fold_dir.mkdir(exist_ok=True, parents=True)

# 划分训练集和验证集
X_train_raw, X_val_raw = X_raw[train_idx], X_raw[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

# 1) SG 平滑
X_train_sg = sg_filter(X_train_raw)
X_val_sg   = sg_filter(X_val_raw)

save_csv(df.iloc[train_idx], X_train_sg, y_train, fold_dir / "train_SG.csv")
save_csv(df.iloc[val_idx],   X_val_sg,   y_val,   fold_dir / "val_SG.csv")

# 2) SNV
mu_train, sigma_train = snv_fit(X_train_sg)
mu_val, sigma_val = snv_fit(X_val_sg)
X_train_snv = snv_transform(X_train_sg, mu_train, sigma_train)
X_val_snv   = snv_transform(X_val_sg,   mu_val, sigma_val)

save_csv(df.iloc[train_idx], X_train_snv, y_train, fold_dir / "train_SNV.csv")
save_csv(df.iloc[val_idx],   X_val_snv,   y_val,   fold_dir / "val_SNV.csv")

# 3) Z-score：训练集统计 -> 训练/验证共用
z_mean, z_std = zscore_fit(X_train_snv)
X_train_z = zscore_transform(X_train_snv, z_mean, z_std)
X_val_z   = zscore_transform(X_val_snv,   z_mean, z_std)

# 4) 保存 CSV
save_csv(df.iloc[train_idx], X_train_z, y_train, fold_dir / "train.csv")
save_csv(df.iloc[val_idx],   X_val_z,   y_val,   fold_dir / "val.csv")

# 5) 保存参数
np.save(fold_dir / "snv_mu.npy",    mu_train)
np.save(fold_dir / "snv_sigma.npy", sigma_train)
np.save(fold_dir / "z_mean.npy",    z_mean)
np.save(fold_dir / "z_std.npy",     z_std)

print(f"Fold 1 done → {fold_dir}")

print("All folds saved under:", OUT_DIR.resolve())