"""
test_single_model_fullpath_v2.py
控制变量实验结果可视化 —— 三线三色 + 标记同色 / Arial / 清新假日
"""
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path

# --------------------------------------------------
# 0. 全局科研风 + Arial 字体 + 可调控字体倍数
# --------------------------------------------------
FONT_SCALE = 1.2
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 12 * FONT_SCALE,
    "axes.labelsize": 14 * FONT_SCALE,
    "axes.titlesize": 16 * FONT_SCALE,
    "xtick.labelsize": 10 * FONT_SCALE,
    "ytick.labelsize": 12 * FONT_SCALE,
    "legend.fontsize": 10 * FONT_SCALE,
    "figure.titlesize": 16 * FONT_SCALE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# --------------------------------------------------
# 0-1. 清新假日配色方案
# --------------------------------------------------
QING_HOLIDAY = {
    'train': "#6AD1A3",   # 清新绿 - 训练集
    'val':   "#FFA288",   # 清新珊瑚 - 验证集
    'line':  "#7FBDDA"    # 清新蓝 - 参考线
}

# ---------- 1. 路径（按需修改） ----------
MODEL_PATH = Path(
    r"F:\study\manggo_program\Easypilpeline\new1.0\A_Dataset\FX10_and_FX17\Hardness_10_140\outlier_isolationforest\Dataset\FX10_and_FX17_TSS_feature_selection_results\results\Ridge\all_fold_FX10_and_FX17_TSS.pkl")
TRAIN_CSV  = Path(
    r"F:\study\manggo_program\Easypilpeline\new1.0\A_Dataset\FX10_and_FX17\Hardness_10_140\outlier_isolationforest\Dataset\FX10_and_FX17_TSS_feature_selection_results\train_all.csv")
VAL_CSV    = Path(
    r"F:\study\manggo_program\Easypilpeline\new1.0\A_Dataset\FX10_and_FX17\Hardness_10_140\outlier_isolationforest\Dataset\FX10_and_FX17_TSS_feature_selection_results\val_all.csv")

SAVE_PNG   = Path(
    r"F:\study\manggo_program\Easypilpeline\new1.0\G_analysis\predict\scatter\Ridge_train_val_scatter.png")
TRAIN_PRED_CSV = Path(
    r"F:\study\manggo_program\Easypilpeline\new1.0\G_analysis\predict\scatter\Ridge_train_pred.csv")
VAL_PRED_CSV   = Path(
    r"F:\study\manggo_program\Easypilpeline\new1.0\G_analysis\predict\scatter\Ridge_val_pred.csv")
# ------------------------------------------

# ---------- 2. 读模型 ----------
model = joblib.load(MODEL_PATH)

# ---------- 3. 读数据 ----------
def read_xy(csv_path: Path):
    df = pd.read_csv(csv_path)
    x = df.drop(columns=["TSS", "Polygon_ID"]).values
    y = df["TSS"].values
    return x, y, df

X_train, y_train, df_train = read_xy(TRAIN_CSV)
X_val,   y_val,   df_val   = read_xy(VAL_CSV)

# ---------- 4. 预测 ----------
y_train_pred = model.predict(X_train)
y_val_pred   = model.predict(X_val)

# ---------- 5. 指标 ----------
def calc_metrics(y_true, y_pred, split=""):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{split} R²   : {r2:.4f}")
    print(f"{split} RMSE : {rmse:.4f}")
    return r2, rmse

r2_train, rmse_train = calc_metrics(y_train, y_train_pred, "Train")
r2_val,   rmse_val   = calc_metrics(y_val,   y_val_pred,   "Val")

# ---------- 6. 保存预测结果 ----------
pd.DataFrame({"True": y_train, "Pred": y_train_pred}).to_csv(TRAIN_PRED_CSV, index=False)
pd.DataFrame({"True": y_val,   "Pred": y_val_pred}).to_csv(VAL_PRED_CSV,   index=False)

print(f"✅ 训练集预测结果已保存：{TRAIN_PRED_CSV}")
print(f"✅ 验证集预测结果已保存：{VAL_PRED_CSV}")

# ---------- 7. 画散点图（清新假日风格） ----------
fig, ax = plt.subplots(figsize=(6, 6))

# 计算坐标范围用于1:1线
lims = [
    min(y_train.min(), y_val.min(), y_train_pred.min(), y_val_pred.min()),
    max(y_train.max(), y_val.max(), y_train_pred.max(), y_val_pred.max())
]

# 绘制1:1参考线（清新蓝）
ax.plot(lims, lims, color=QING_HOLIDAY['line'], linestyle='--', linewidth=2, alpha=0.8, label='1:1 Line')

# 绘制训练集散点（清新绿，圆形标记）
ax.scatter(y_train, y_train_pred,
           c=QING_HOLIDAY['train'],
           marker='o',
           s=80,
           alpha=0.7,
           edgecolors='white',
           linewidth=0.5,
           label=f"Train  R²={r2_train:.3f}, RMSE={rmse_train:.2f} °Brix")

# 绘制验证集散点（清新珊瑚，三角形标记）
ax.scatter(y_val, y_val_pred,
           c=QING_HOLIDAY['val'],
           marker='^',
           s=80,
           alpha=0.7,
           edgecolors='white',
           linewidth=0.5,
           label=f"Test   R²={r2_val:.3f}, RMSE={rmse_val:.2f} °Brix")

# 设置标签和标题
ax.set_xlabel("True TSS", fontweight='bold')
ax.set_ylabel("Predicted TSS", fontweight='bold')
ax.set_title("Ridge | Train vs Test", fontweight='bold')

# 设置坐标轴范围相等
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal', adjustable='box')

# 添加图例（白色背景，带边框）
legend = ax.legend(loc='upper left', frameon=True, edgecolor='gray', facecolor='white', framealpha=0.9)
legend.get_frame().set_linewidth(1.0)

# 添加网格线（淡灰色）
ax.grid(True, linestyle='--', alpha=0.3, color='gray')

# 移除顶部和右侧边框（spines）
sns.despine(ax=ax, top=True, right=True)

plt.tight_layout()

SAVE_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(SAVE_PNG, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✅ 散点图已保存：{SAVE_PNG}")