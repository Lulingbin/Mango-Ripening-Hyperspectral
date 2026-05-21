"""
train_any_featureset.py
一次完成：
  1) 多个特征子集（union / PCA4FS / RF4FS …）
  2) 5 折交叉
  3) 多模型（RF / XGB / Ridge / Lasso / PLSR / SLP / MLP）
"""

# ---------- 环境 ----------
import os, json, joblib, warnings, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
# from sklearn.ensemble import RandomForestRegressor
# # from sklearn.linear_model import Ridge, Lasso
# from sklearn.cross_decomposition import PLSRegression

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 路径 ----------
sys.path.append(str(Path(__file__).resolve().parent.parent / "E_models"))

from regressors import *   # XGBRegressor 等自定义模型


# ---------- 工具 ----------
def Rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

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
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# ---------- PyTorch 数据集 ----------
class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ---------- 单隐藏层感知机 ----------
class SLPRegressorTorch(BaseEstimator, RegressorMixin):
    def __init__(self, neurons=50, epochs=100, lr=0.01,
                 batch_size=256, random_state=42,patience=20):
        self.neurons, self.epochs, self.lr = neurons, epochs, lr
        self.batch_size, self.random_state = batch_size, random_state
        self.net_ = None
        self.patience = patience  # 新增
    def _build_net(self, n_features):
        torch.manual_seed(self.random_state)
        return nn.Sequential(
            nn.Linear(n_features, self.neurons), nn.ReLU(),
            nn.Linear(self.neurons, 1)
        ).to(DEVICE)
    def fit(self, X, y, X_val=None, y_val=None):   # 新增参数
        X, y = X.astype(np.float32), y.astype(np.float32)
        self.net_ = self._build_net(X.shape[1])
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        loader_tr = torch.utils.data.DataLoader(
            TorchDataset(X, y), batch_size=self.batch_size, shuffle=True)

        # 如果有外部验证集，就用它做早停
        if X_val is not None and y_val is not None:
            X_val = torch.tensor(X_val.astype(np.float32)).to(DEVICE)
            y_val = torch.tensor(y_val.astype(np.float32)).to(DEVICE)

        early = EarlyStopping(patience=self.patience)
        self.net_.train()
        for epoch in range(self.epochs):
            for xb, yb in loader_tr:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(self.net_(xb), yb)
                loss.backward()
                opt.step()

            # ---- 用验证集计算 loss ----
            if X_val is not None:
                self.net_.eval()
                with torch.no_grad():
                    val_loss = loss_fn(self.net_(X_val), y_val.unsqueeze(1)).item()
                early(val_loss)
                if early.early_stop:  # 原来是 early.stop
                    break
                self.net_.train()

        return self
    def predict(self, X):
        self.net_.eval()
        with torch.no_grad():
            X = torch.tensor(X.astype(np.float32)).to(DEVICE)
            return self.net_(X).cpu().numpy().ravel()

# ---------- 多隐藏层感知机 ----------
class MLPRegressorTorch(BaseEstimator, RegressorMixin):
    def __init__(self, layers=2, neurons=50, epochs=100, lr=0.05,
                 batch_size=256, random_state=42,patience = 20):
        self.layers, self.neurons, self.epochs, self.lr = layers, neurons, epochs, lr
        self.batch_size, self.random_state = batch_size, random_state
        self.net_ = None
        self.patience = patience
    def _build_net(self, n_features):
        torch.manual_seed(self.random_state)
        modules = [nn.Linear(n_features, self.neurons), nn.ReLU()]
        for _ in range(self.layers - 1):
            modules += [nn.Linear(self.neurons, self.neurons), nn.ReLU()]
        modules.append(nn.Linear(self.neurons, 1))
        return nn.Sequential(*modules).to(DEVICE)

    def fit(self, X, y, X_val=None, y_val=None):  # 新增参数
        X, y = X.astype(np.float32), y.astype(np.float32)
        self.net_ = self._build_net(X.shape[1])
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        loader_tr = torch.utils.data.DataLoader(
            TorchDataset(X, y), batch_size=self.batch_size, shuffle=True)

        # 如果有外部验证集，就用它做早停
        if X_val is not None and y_val is not None:
            X_val = torch.tensor(X_val.astype(np.float32)).to(DEVICE)
            y_val = torch.tensor(y_val.astype(np.float32)).to(DEVICE)

        early = EarlyStopping(patience=self.patience)
        self.net_.train()
        for epoch in range(self.epochs):
            for xb, yb in loader_tr:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(self.net_(xb), yb)
                loss.backward()
                opt.step()

            # ---- 用验证集计算 loss ----
            if X_val is not None:
                self.net_.eval()
                with torch.no_grad():
                    val_loss = loss_fn(self.net_(X_val), y_val.unsqueeze(1)).item()
                early(val_loss)
                if early.early_stop:  # 原来是 early.stop
                    break
                self.net_.train()

        return self
    def predict(self, X):
        self.net_.eval()
        with torch.no_grad():
            X = torch.tensor(X.astype(np.float32)).to(DEVICE)
            return self.net_(X).cpu().numpy().ravel()










def train_FXN(target_X,
              target_Y,
              raw_dir = Path(r"F:\study\manggo_program\Easypilpeline\new1.0\A_Dataset\FX10_and_FX17\Hardness_10_140\outlier_isolationforest\Dataset\fold_1\FX10_and_FX17")):
    # =========================================================
    # ========= 用户配置区（只改下面 3 个变量即可） ===========
    # =========================================================

    INPUT_DIR      = raw_dir /f"{target_X}_{target_Y}_feature_selection_results"
    RESULTS      = raw_dir /f"{target_X}_{target_Y}_feature_selection_results"/ "results"
    PLOT_DIR     = RESULTS /"plots"
    FEATURE_SETS = ["all","union", "PCA4FS", "RF4FS", "SLP4FS", "GA4FS"]
    # =========================================================

    RESULTS.mkdir(exist_ok=True, parents=True)
    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    RANDOM_STATE = 42


    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    # ---------- 统一超参数网格 ----------
    MODEL_GRID = {
        "RF":   (RandomForestRegressor,
                  {"n_estimators":[100,300,500], "max_depth":[5,10,20]}),
        "XGB":  (XGBRegressor,
                  {"n_estimators":[100,300,500], "max_depth":[3,6,9], "learning_rate":[0.01,0.05]}),
        "Ridge":(Ridge, {"alpha":[0.01,0.1,1,10,100]}),
        "Lasso":(Lasso, {"alpha":[0.0001,0.001,0.01,0.1,1], "max_iter":[10000]}),
        "PLSR": (PLSRegression, {"n_components":[2,4,6,8,10]}),
        "SLP":  (SLPRegressorTorch,
                  {"epochs":[ 150 ], "neurons":[10, 20, 50, 100, 200, 500], "lr":[0.001],"batch_size":[32]}),
        "MLP":  (MLPRegressorTorch,
                  {"epochs":[150], "layers":[2,3,4], "neurons":[10, 20, 50, 100, 200, 500], "lr":[0.001],"batch_size":[32]}),

    }

    # ---------- 主流程 ----------
    from collections import defaultdict

    model_order = list(MODEL_GRID.keys())
    score_bank  = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    all_cv_rows = []        # 收集所有 GridSearchCV 结果

    for feat_suffix in FEATURE_SETS:
        fold_dir  = INPUT_DIR
        train_csv = fold_dir / f"train_{feat_suffix}.csv"
        val_csv   = fold_dir / f"val_{feat_suffix}.csv"
        if not (train_csv.exists() and val_csv.exists()):
            print(f"⚠️ 跳过 {feat_suffix} fold{target_X}_{target_Y}（CSV 缺失）")
            continue

        train_df = pd.read_csv(train_csv)
        val_df   = pd.read_csv(val_csv)
        X_train  = train_df.drop(columns=[target_Y,'Polygon_ID']).values
        y_train  = train_df[target_Y].values
        X_val    = val_df.drop(columns=[target_Y,'Polygon_ID']).values
        y_val    = val_df[target_Y].values

        for model_name, (model_cls, grid) in MODEL_GRID.items():
            model_dir = RESULTS / model_name
            model_dir.mkdir(exist_ok=True, parents=True)

            est = model_cls()
            cv  = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            gs  = GridSearchCV(est, grid, cv=cv, scoring="r2", n_jobs=6)
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            # 如果是 PyTorch 模型，再 fit 一次，用验证集做早停
            if model_name in {"SLP", "MLP",}:
                best.fit(X_train, y_train, X_val=X_val, y_val=y_val)

            # ---------- 收集超参数搜索结果 ----------
            cv_df = (pd.DataFrame(gs.cv_results_)
                       .filter(regex='^param_|mean_test_score|std_test_score')
                       .rename(columns=lambda c: c.replace('param_', '')))
            cv_df.insert(0, 'feature_set', feat_suffix)
            cv_df.insert(1, 'fold', 1)
            cv_df.insert(2, 'model', model_name)
            cv_df.insert(3, 'best_mean_score', gs.best_score_)
            cv_df.insert(4, 'best_std_score',
                         cv_df.loc[gs.best_index_, 'std_test_score'])
            all_cv_rows.append(cv_df)

            # 预测
            y_pred_tr  = best.predict(X_train)
            y_pred_val = best.predict(X_val)

            # 指标
            r2_tr   = r2_score(y_train, y_pred_tr)
            rmse_tr = Rmse(y_train, y_pred_tr)
            r2_val  = r2_score(y_val, y_pred_val)
            rmse_v  = Rmse(y_val, y_pred_val)

            # 保存模型 & 预测 CSV
            pre = f"{feat_suffix}_fold_{target_X}_{target_Y}"
            joblib.dump(best, model_dir / f"{pre}.pkl")
            pd.DataFrame({"fold": 1, "true": y_val, "pred": y_pred_val}).to_csv(
                model_dir / f"{pre}_pred.csv", index=False)
            with open(model_dir / f"{pre}_best_params.json", "w") as f:
                json.dump(gs.best_params_, f, indent=2, ensure_ascii=False)

            # 训练集散点图
            plt.figure(figsize=(4, 4))
            sns.scatterplot(x=y_train, y=y_pred_tr, alpha=0.7, color="orange")
            plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
            plt.title(f"{model_name} | {feat_suffix} | {target_X}_{target_Y} (train)\nR²={r2_tr:.3f}")
            plt.xlabel("True"); plt.ylabel("Pred")
            plt.tight_layout()
            plt.savefig(model_dir / f"{pre}_train_scatter.png", dpi=300)
            plt.close()

            # 验证集散点图
            plt.figure(figsize=(4, 4))
            sns.scatterplot(x=y_val, y=y_pred_val, alpha=0.7)
            plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
            plt.title(f"{model_name} | {feat_suffix} | {target_X}_{target_Y} (val)\nR²={r2_val:.3f}")
            plt.xlabel("True"); plt.ylabel("Pred")
            plt.tight_layout()
            plt.savefig(model_dir / f"{pre}_val_scatter.png", dpi=300)
            plt.close()

            # 记录到汇总库
            score_bank['r2'][1][feat_suffix][model_name]   = r2_val
            score_bank['rmse'][1][feat_suffix][model_name] = rmse_v
            print(f"{feat_suffix}_fold_{1}_{model_name} ✓")

    # ---------- 写出统一超参数总表 ----------
    tuning_root = RESULTS / "tuning_summary"
    tuning_root.mkdir(exist_ok=True, parents=True)
    tuning_all_df = pd.concat(all_cv_rows, ignore_index=True)
    tuning_all_df.to_csv(tuning_root / "tuning_summary_all.csv", index=False)
    print("✅ 所有超参数搜索结果已合并到 tuning_summary_all.csv")

    # 看看每一折到底录了多少条记录
    for fold in range(1, 2):
        print(f"fold{fold} 实际记录：r2={len(score_bank['r2'][fold])}, "
              f"rmse={len(score_bank['rmse'][fold])}")


    # ---------- 生成 summary 文件夹 ----------
    summary_root = RESULTS / "summary"
    summary_root.mkdir(exist_ok=True, parents=True)
    all_dir = summary_root / "all"
    all_dir.mkdir(exist_ok=True, parents=True)

    # --------------------------------------------------
    # 1) 长表：5 折 + 平均  →  all_r2.csv / all_rmse.csv
    # --------------------------------------------------
    rows_r2 = []
    rows_rmse = []
    for fold in range(1, 2):
        for feat in FEATURE_SETS:
            for mdl in model_order:
                r2   = score_bank['r2'][fold].get(feat, {}).get(mdl, np.nan)
                rmse = score_bank['rmse'][fold].get(feat, {}).get(mdl, np.nan)
                rows_r2.append({'Fold': fold, 'FeatureSet': feat, 'Model': mdl, 'R2': r2})
                rows_rmse.append({'Fold': fold, 'FeatureSet': feat, 'Model': mdl, 'RMSE': rmse})

    # 追加平均行
    for feat in FEATURE_SETS:
        for mdl in model_order:
            mean_r2   = np.nanmean([score_bank['r2'][f][feat].get(mdl, np.nan) for f in range(1, 6)])
            mean_rmse = np.nanmean([score_bank['rmse'][f][feat].get(mdl, np.nan) for f in range(1, 6)])
            rows_r2.append({'Fold': 'Mean', 'FeatureSet': feat, 'Model': mdl, 'R2': mean_r2})
            rows_rmse.append({'Fold': 'Mean', 'FeatureSet': feat, 'Model': mdl, 'RMSE': mean_rmse})

    pd.DataFrame(rows_r2).to_csv(all_dir / "all_r2.csv", index=False)
    pd.DataFrame(rows_rmse).to_csv(all_dir / "all_rmse.csv", index=False)

    # --------------------------------------------------
    # 2) 每折一张热力图  (FeatureSet × Model)
    # --------------------------------------------------
    for fold in range(1, 2):
        fold_r2 = pd.DataFrame({feat: {m: score_bank['r2'][fold].get(feat, {}).get(m, np.nan)
                                       for m in model_order}
                                for feat in FEATURE_SETS})

        # 固定宽度 8 英寸，高度按特征数自适应
        plt.figure(figsize=(8, max(3, len(FEATURE_SETS) * 0.5)))
        sns.heatmap(fold_r2, annot=True, fmt=".3f", cmap="YlGnBu",
                    vmin=0, vmax=1, cbar_kws={'shrink': .7})
        plt.title(f"Fold {fold}  R²")
        plt.ylabel("Feature Set")
        plt.xlabel("Model")
        plt.xticks(rotation=15, ha='right')   # 文字略倾斜即可
        plt.tight_layout()
        plt.savefig(all_dir / f"heatmap_r2_fold{fold}.png", dpi=300)
        plt.close()

    # --------------------------------------------------
    # 3) 平均值热力图（与原脚本一致）
    # --------------------------------------------------
    avg_r2 = pd.DataFrame({feat: {m: np.nanmean([score_bank['r2'][f][feat].get(m, np.nan)
                                                 for f in range(1, 6)])
                                  for m in model_order}
                           for feat in FEATURE_SETS})

    plt.figure(figsize=(8, max(3, len(FEATURE_SETS) * 0.5)))
    sns.heatmap(avg_r2, annot=True, fmt=".3f", cmap="YlGnBu",
                vmin=0, vmax=1, cbar_kws={'shrink': .7})
    plt.title("Average R²")
    plt.ylabel("Feature Set")
    plt.xlabel("Model")
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(all_dir / "heatmap_r2.png", dpi=300)
    plt.close()

    print("🎉 每折及平均热力图已生成：", all_dir)


# ALL_Y = ["Hardness","TSS","carotenoids","starch_content"]
# ALL_Y = ["carotenoids","starch_content"]
# ALL_X = ["FX10_and_FX17"]
#
#
# for xi in ALL_X:
#     for yj in ALL_Y:
#         train_FXN(xi,yj,raw_dir = Path(r"F:\study\manggo_program\Easypilpeline\new1.0\A_Dataset\FX10_and_FX17\Hardness_10_140\outlier_isolationforest\Dataset"))
#
#
# ALL_Y = ["Hardness","TSS","carotenoids","starch_content"]
# ALL_X = ["FX10"]
# for xi in ALL_X:
#     for yj in ALL_Y:
#         train_FXN(xi,yj,raw_dir = Path(r"F:\study\manggo_program\Easypilpeline\new1.0\A_Dataset\FX10\Hardness_10_140\outlier_isolationforest\Dataset"))

ALL_Y = ["carotenoids","starch_content"]
ALL_X = ["FX17"]
for xi in ALL_X:
    for yj in ALL_Y:
        train_FXN(xi,yj,raw_dir = Path(r"F:\study\manggo_program\Easypilpeline\new1.0\A_Dataset\FX17\Hardness_10_140\outlier_isolationforest\Dataset"))
