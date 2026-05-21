"""
test_single_model_fullpath.py
"""
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path

# ---------- 1. 直接填 3 个绝对路径 ----------
MODEL_PATH = (
    Path(r"F:\study\manggo_program\PLSR_pipeline\A_Dataset\FX10\700\Hardness_10_140\outlier_isolationforest\clean_csv_split_5fold\results\Ridge\all_fold_1.pkl"))
TEST_CSV   =\
    Path(r"F:\study\manggo_program\PLSR_pipeline\A_Dataset\FX10\700\Hardness_10_140\outlier_isolationforest\clean_csv_split_5fold\test\test_PCA4FS.csv")
SAVE_PNG   =\
    Path(r"F:\study\manggo_program\PLSR_pipeline\A_Dataset\FX10\700\Hardness_10_140\outlier_isolationforest\clean_csv_split_5fold\results\test_scatter\PCA4FS_fold3_MLP_test.png")
# ------------------------------------------

# ---------- 2. 读模型 ----------
model = joblib.load(MODEL_PATH)

# ---------- 3. 读测试集 ----------
df = pd.read_csv(TEST_CSV)
X_test = df.drop(columns=["Hardness"]).values
y_test = df["Hardness"].values

# ---------- 4. 预测 ----------
y_pred = model.predict(X_test)

# ---------- 5. 指标 ----------
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test R²   : {r2:.4f}")
print(f"Test RMSE : {rmse:.4f}")

# ---------- 6. 画散点图 ----------
plt.figure(figsize=(4.5, 4))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, 'r--')
plt.title(f"MLP | PCA4FS | Fold 3 (test)\nR²={r2:.3f}")
plt.xlabel("True Hardness")
plt.ylabel("Predicted Hardness")
plt.tight_layout()
SAVE_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(SAVE_PNG, dpi=300)
plt.close()
print(f"✅ 散点图已保存：{SAVE_PNG}")