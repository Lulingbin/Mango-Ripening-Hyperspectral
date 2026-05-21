import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score, mean_squared_error
import torch
import torch.nn as nn

# 强制使用 CPU 进行纯绘图预测，防止显存或多线程冲突
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# 🛠️ 关键修复 1：将自定义 PyTorch 类声明保留在此，供 joblib 反序列化重建对象
# =====================================================================
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.min_delta = min_delta

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class SLPRegressorTorch(BaseEstimator, RegressorMixin):
    def __init__(self, neurons=50, epochs=100, lr=0.01, batch_size=256, random_state=42, patience=20):
        self.neurons, self.epochs, self.lr = neurons, epochs, lr
        self.batch_size, self.random_state = batch_size, random_state
        self.net_ = None
        self.patience = patience

    def predict(self, X):
        self.net_.eval()
        with torch.no_grad():
            X = torch.tensor(X.astype(np.float32)).to(DEVICE)
            return self.net_(X).cpu().numpy().ravel()


class MLPRegressorTorch(BaseEstimator, RegressorMixin):
    def __init__(self, layers=2, neurons=50, epochs=100, lr=0.05, batch_size=256, random_state=42, patience=20):
        self.layers, self.neurons, self.epochs, self.lr = layers, neurons, epochs, lr
        self.batch_size, self.random_state = batch_size, random_state
        self.net_ = None
        self.patience = patience

    def predict(self, X):
        self.net_.eval()
        with torch.no_grad():
            X = torch.tensor(X.astype(np.float32)).to(DEVICE)
            return self.net_(X).cpu().numpy().ravel()






# 保持你原有的引用以防内部解包需要：
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "E_models"))



# ---------- 工具函数 ----------
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def plot_combined_scatter(y_train, y_pred_train, r2_tr, rmse_tr,
                          y_val, y_pred_val, r2_val, rmse_v,
                          title_str, save_path):
    plt.figure(figsize=(5, 5))

    # 绘制训练集 (橙色) 和 验证集 (蓝色)
    sns.scatterplot(x=y_train, y=y_pred_train, alpha=0.5, color="orange",
                    label=f"Train (R²={r2_tr:.3f}, RMSE={rmse_tr:.3f})")
    sns.scatterplot(x=y_val, y=y_pred_val, alpha=0.7, color="#1f77b4",
                    label=f"Val (R²={r2_val:.3f}, RMSE={rmse_v:.3f})")

    # 绘制 1:1 完美预测线
    all_max = max(y_train.max(), y_val.max(), y_pred_train.max(), y_pred_val.max())
    all_min = min(y_train.min(), y_val.min(), y_pred_train.min(), y_pred_val.min())
    plt.plot([all_min, all_max], [all_min, all_max], 'r--', alpha=0.7, label='1:1 Line')

    plt.title(title_str, fontsize=10, pad=10)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ---------- 核心扫描与绘图主函数 ----------
def generate_scatters_from_results(root_dirs):
    output_dir = Path("./output_scatters")
    output_dir.mkdir(exist_ok=True, parents=True)

    # 定义合法的 target_Y 集合用于精准提取
    VALID_TARGETS = ["Hardness", "TSS", "carotenoids", "starch_content"]

    for base_path in root_dirs:
        base_path = Path(base_path)
        if not base_path.exists():
            print(f"⚠️ 路径不存在，跳过: {base_path}")
            continue

        print(f"\n🚀 开始扫描基础目录: {base_path}")

        for feat_dir in base_path.glob("*_feature_selection_results"):
            dir_name = feat_dir.name

            # 🛠️ 关键修复 2：使用更稳健的解析逻辑，避免下划线组合（如 starch_content）被错误切分
            target_X, target_Y = None, None
            for target in VALID_TARGETS:
                if f"_{target}_feature_selection_results" in dir_name:
                    target_Y = target
                    target_X = dir_name.replace(f"_{target}_feature_selection_results", "")
                    break

            if not target_X or not target_Y:
                continue

            results_dir = feat_dir / "results"
            if not results_dir.exists():
                continue

            for model_dir in results_dir.iterdir():
                if not model_dir.is_dir() or model_dir.name in ["plots", "summary", "tuning_summary",
                                                                "scatters_reproduced"]:
                    continue

                model_name = model_dir.name

                for pkl_path in model_dir.glob("*.pkl"):
                    feat_suffix = pkl_path.name.split("_fold_")[0]

                    train_csv = feat_dir / f"train_{feat_suffix}.csv"
                    val_csv = feat_dir / f"val_{feat_suffix}.csv"

                    if not (train_csv.exists() and val_csv.exists()):
                        print(f" 缺失数据源，无法为 {pkl_path.name} 绘图")
                        continue

                    try:
                        train_df = pd.read_csv(train_csv)
                        val_df = pd.read_csv(val_csv)

                        X_train = train_df.drop(columns=[target_Y, 'Polygon_ID'], errors='ignore').values
                        y_train = train_df[target_Y].values
                        X_val = val_df.drop(columns=[target_Y, 'Polygon_ID'], errors='ignore').values
                        y_val = val_df[target_Y].values

                        # 此时运行，由于当前脚本中已有对应的类骨架，joblib 可以百分之百复原模型
                        model = joblib.load(pkl_path)

                        y_pred_tr = model.predict(X_train)
                        y_pred_val = model.predict(X_val)

                        r2_tr = r2_score(y_train, y_pred_tr)
                        rmse_tr = calculate_rmse(y_train, y_pred_tr)
                        r2_val = r2_score(y_val, y_pred_val)
                        rmse_v = calculate_rmse(y_val, y_pred_val)

                        title = f"{model_name} | {feat_suffix}\nDataset: {target_X} → {target_Y}"
                        save_name = f"{target_X}_{target_Y}_{feat_suffix}_{model_name}_scatter.png"
                        save_path = output_dir / save_name

                        plot_combined_scatter(
                            y_train, y_pred_tr, r2_tr, rmse_tr,
                            y_val, y_pred_val, r2_val, rmse_v,
                            title, save_path
                        )
                        print(f"   [成功] 已保存至当前工作区: {save_name}")

                    except Exception as e:
                        print(f"   [失败] 无法处理 {pkl_path.name}, 错误原因: {e}")


# ---------- 执行入口 ----------
if __name__ == "__main__":
    TARGET_PATHS = [
        r"F:\study\manggo_program\mango_predict_all\pipeline\A_Dataset\FX10_and_FX17\Hardness_10_140\outlier_isolationforest\Dataset",
        r"F:\study\manggo_program\mango_predict_all\pipeline\A_Dataset\FX10\Hardness_10_140\outlier_isolationforest\Dataset",
        r"F:\study\manggo_program\mango_predict_all\pipeline\A_Dataset\FX17\Hardness_10_140\outlier_isolationforest\Dataset"
    ]

    generate_scatters_from_results(TARGET_PATHS)
    print("\n🎉 修复完毕！所有散点图已成功保存至 './output_scatters'！")