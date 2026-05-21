import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn_genetic import GAFeatureSelectionCV
from sklearn_genetic.space import Integer
import warnings, os, sys
from sklearn.model_selection import GridSearchCV   # 这里加上
import logging
from typing import Optional
import inspect
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


TOP_K = int(10)
RANDOM_STATE = 42

# ---------- 特征选择函数 ----------

############pca4fs
def pca4fs(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,      # y 仅占位，保持接口一致
    k: int = TOP_K,
    var_keep: float = 0.99
) -> np.ndarray:
    """
    PCA4FS 特征选择（与 Cruz-Tirado et al. 2022 原文对齐版）

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_wavelengths)
        光谱矩阵
    y : None
        占位，方法本身无监督
    k : int
        最终要返回的波长个数
    var_keep : float
        累积解释方差需达到的比例，如 0.99

    Returns
    -------
    selected_idx : ndarray, shape (k,)
        选出的波长索引，按重要性降序排列
    """
    # 1. 数据检查
    if np.any(~np.isfinite(X)):
        raise ValueError("Input contains NaN/Inf.")

    # 2. 计算 PCA
    pca = PCA().fit(X)

    # 3. 选出累积解释方差 < var_keep 的主成分
    cum_var = pca.explained_variance_ratio_.cumsum()
    n_pcs = int(np.searchsorted(cum_var, var_keep, side='right'))
    if n_pcs == 0:
        raise ValueError("var_keep too small, no PCs retained.")

    # 4. 在每个 PC 内取 |loading| 最大的前 k_p 个波长
    load_abs = np.abs(pca.components_[:n_pcs])      # (n_pcs, n_wl)
    topk_per_pc = min(k, load_abs.shape[1])
    # argpartition 返回前 topk_per_pc 的索引
    idx_top = np.argpartition(-load_abs, topk_per_pc - 1, axis=1)[:, :topk_per_pc]

    # 5. 综合得分：累加出现的绝对 loading
    score = np.zeros(load_abs.shape[1])
    for pc_idx in range(n_pcs):
        for wl_idx in idx_top[pc_idx]:
            score[wl_idx] += load_abs[pc_idx, wl_idx]

    # 6. 返回得分最高的前 k 个波长索引
    selected_idx = np.argsort(-score)[:k]
    print(f"得分最高的前 k 个波长索引{selected_idx}")
    return selected_idx


############rf4fs
def rf4fs(X, y, k=TOP_K):
    mdl = RandomForestRegressor(
        n_estimators=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    mdl.fit(X, y)
    imp = mdl.feature_importances_
    thresh = imp.mean()
    idx = np.where(imp > thresh)[0]
    idx = idx[np.argsort(imp[idx])[-k:]] if len(idx) > k else idx
    print(f"得分最高的前 k 个波长索引{idx}")
    return idx

def slp4fs_gedeon(X, y, k=TOP_K, random_state=RANDOM_STATE):
    """
    回归任务下的 SLP4FS-Gedeon 特征选择（带打印进度）。
    参考：Gedeon, T. D. (1997). Data Mining and Knowledge Discovery, 1(2), 153-161.
    """
    n_trials = len(range(1, 101, 5))
    print("=== 1. 网格搜索最优隐层神经元数（1~100，步长 10）===")
    best_mdl, best_score = None, -np.inf
    for idx, nh in enumerate(range(1, 101, 5), 1):
        print(f"[{idx:2d}/{n_trials}] 训练 nh={nh:2d} ... ", end="")
        mdl = MLPRegressor(
            hidden_layer_sizes=(nh, nh),
            activation='tanh',
            max_iter=1000,
            random_state=random_state
        )
        mdl.fit(X, y)
        score = mdl.score(X, y)
        print(f"R²={score:.4f}", end="")
        if score > best_score:
            best_score, best_mdl = score, mdl
            print("  <-- 新最佳")
        else:
            print()

    print(f"网格搜索完成，最佳 nh={best_mdl.hidden_layer_sizes[0]}，最佳 R²={best_score:.4f}\n")

    # 2. 取出权重
    W0, W1 = best_mdl.coefs_[:2]
    n_features, nh = W0.shape
    contrib = np.zeros(n_features)

    print("=== 2. Gedeon 贡献度计算 ===")
    for i in range(n_features):
        print(f"\r计算第 {i+1:4d}/{n_features} 个特征贡献度 ...", end="", flush=True)
        for j in range(nh):
            if W0[i, j] == 0:
                continue
            p_ij = abs(W0[i, j]) / np.sum(np.abs(W0[:, j]))
            for l in range(nh):
                p_jk = abs(W1[j, l]) / np.sum(np.abs(W1[j, :]))
                contrib[i] += p_ij * p_jk
    print("\n贡献度计算完成！\n")

    idx = np.argsort(contrib)[::-1][:k]
    print("=== 3. 结果 ===")
    print(f"Top-{k} 特征索引（按贡献降序）：{idx.tolist()}")
    return idx



###########ga4fs
# ---------------- 自定义回调：打印每一代的日志 ----------------
class PrintProgressCallback:
    def __init__(self):
        self.gen = 0
    def __call__(self, ga_instance, fitness):
        self.gen += 1
        print(f"[Gen {self.gen:3d}] best: {max(fitness):8.4f}  mean: {np.mean(fitness):8.4f}")


def ga4fs(X, y, k=TOP_K):
    from sklearn_genetic.callbacks import ConsecutiveStopping
    early_stop = ConsecutiveStopping(generations=10, metric='fitness')

    ga = GAFeatureSelectionCV(
        estimator=RandomForestRegressor(300, random_state=RANDOM_STATE, n_jobs=-1),
        cv=3,
        scoring='neg_root_mean_squared_error',
        population_size=10,
        generations=100,
        max_features=X.shape[1] - 1,
        verbose=1,
        n_jobs=-1
    )
    print(">>> GA 开始进化 ...")
    ga.fit(X, y, callbacks=[early_stop])
    print(">>> GA 进化完成")

    mask = ga.support_                      # 长度 = 原始特征数
    selected_idx = np.where(mask)[0]        # 被选中的原始列号

    # 随机森林只在被选中的特征上训练，所以长度 = len(selected_idx)
    importances = ga.best_estimator_.feature_importances_

    # 将重要性排序并取前 k 个
    top_k_idx_in_selected = np.argsort(importances)[::-1][:k]
    top_k_original_idx = selected_idx[top_k_idx_in_selected]

    print(f"\n>>> GA 选出的 Top-{k} 特征索引: {top_k_original_idx}\n")
    return top_k_original_idx






methods = {
    'PCA4FS': pca4fs,
    'RF4FS': rf4fs,
    'SLP4FS': slp4fs_gedeon,
     'GA4FS': ga4fs,

}



def select_fs(target_X,target_Y,):
    fold_dir = OUT_DIR / f"fold_1"
    result_dir = fold_dir / f"{target_X}_{target_Y}_feature_selection_results"
    result_dir.mkdir(exist_ok=True, parents=True)

    train_csv = fold_dir / "train.csv"
    df = pd.read_csv(train_csv)
    X = df.filter(regex="^Band_").values
    # X_first_224 = X[:, :224]
    # X_laster_112 = X[:, 224:]
    #
    # if target_X == "FX10":
    #     X=X_first_224
    # elif target_X == "FX17":
    #     X=X_laster_112
    # elif target_X == "FX10_and_FX17":
    #     X=X

    # print(df.columns)
    # print(target_Y)
    y = df[target_Y].values.astype(int)  # 分类标签

    for name, func in methods.items():
        idx = func(X, y, TOP_K)
        np.save(result_dir / f"{name}_top{TOP_K}.npy", idx)

        # 确保列名格式正确
        wavelength_nm = []
        band_columns = df.filter(regex="^Band_").columns  # 获取所有以 "Band_" 开头的列名
        for i in idx:
            col_name = band_columns[i]  # 直接使用特征列名
            if col_name.startswith("Band_"):
                try:
                    wavelength = int(col_name.split('_')[-1])
                    wavelength_nm.append(wavelength)
                except ValueError:
                    raise ValueError(f"列名 {col_name} 不符合预期格式 'Band_<wavelength>'")
            else:
                raise ValueError(f"列名 {col_name} 不符合预期格式 'Band_<wavelength>'")

        pd.DataFrame({
            "method": name,
            "wavelength_index": idx,
            "wavelength_nm": wavelength_nm,
        }).to_csv(result_dir / f"{name}_top{TOP_K}.csv", index=False)
        print(f"Fold  | {name} → {len(idx)} features saved.")

    print("All feature selection results saved under each fold's 'feature_selection_results' folder.")


# ---------- 用户配置 ----------
OUT_DIR = Path(r"F:\study\manggo_program\Easypilpeline\new1.0\A_Dataset\FX10_and_FX17\Hardness_10_140\outlier_isolationforest\Dataset")


# --------------------------------


ALL_Y = ["Hardness","TSS","carotenoids","starch_content"]
ALL_X = ["FX10_and_FX17"]
ALL_X1 = ["FX10"]
ALL_X2 = ["FX17"]



# for xi in ALL_X:
#     for yj in ALL_Y:
#         select_fs(xi,yj)

OUT_DIR = Path(r"F:\study\manggo_program\Easypilpeline\new1.0\A_Dataset\FX10\Hardness_10_140\outlier_isolationforest\Dataset")
for xi in ALL_X1:
    for yj in ALL_Y:
        select_fs(xi,yj)

OUT_DIR = Path(r"F:\study\manggo_program\Easypilpeline\new1.0\A_Dataset\FX17\Hardness_10_140\outlier_isolationforest\Dataset")
for xi in ALL_X2:
    for yj in ALL_Y:
        select_fs(xi,yj)