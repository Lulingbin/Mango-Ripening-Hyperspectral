# outlier_isolationforest.py
import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'F:\study\manggo_program\mango_predict_all\pipeline\A_Dataset\FX10\Hardness_10_140\FX10_all_Hardness_10_140.csv')
dpi = 300
id_col = 'Polygon_ID'
feat_cols = [c for c in df.columns if c.startswith('Band_')]

X = df[feat_cols].values
clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
mask = clf.fit_predict(X) == 1   # 1 = normal, -1 = outlier

clean = df[mask].reset_index(drop=True)
removed = df[~mask].reset_index(drop=True)
removed_ids = removed[id_col].tolist()

out_dir = 'outlier_isolationforest'
os.makedirs(out_dir, exist_ok=True)
clean.to_csv(os.path.join(out_dir, 'clean.csv'), index=False)
removed.to_csv(os.path.join(out_dir, 'removed.csv'), index=False)
pd.Series(removed_ids, name='Removed_Polygon_ID').to_csv(
    os.path.join(out_dir, 'removed_ids.csv'), index=False)

wave = np.arange(len(feat_cols))
colors = ['lightgray' if m else 'red' for m in mask]
dpi, height_inch = 300, 4 * len(df) / dpi
fig, ax = plt.subplots(figsize=(10, height_inch), dpi=dpi)
for curve, c in zip(df[feat_cols].values, colors):
    ax.plot(wave, curve, color=c, linewidth=0.4)
ax.set_title('IsolationForest: gray=normal, red=outlier')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'plot.png'), dpi=dpi)
plt.close()
print(f'IsolationForest done → {out_dir}')


# 1. 如果是高维数据，先用 PCA 压到 2D 方便画图


pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)

plt.figure(figsize=(5,4), dpi=300)
scatter = plt.scatter(
    X_2d[:, 0], X_2d[:, 1],
    c=np.where(mask, 0, 1),        # 0 正常，1 异常
    cmap='coolwarm', s=8, alpha=0.8
)
plt.title('IsolationForest result (PCA view)')
# 改为（不再警告）
legend_elements = scatter.legend_elements()[0]
plt.legend(legend_elements, ['Normal', 'Outlier'])
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'isoforest_pca.png'))
plt.close()

# ========== 追加：平均路径长度可视化 ==========
# 1. 提取平均路径长度
avg_path = clf.decision_function(X) * -1   # IF 返回负距离，取反即越大越异常
sort_idx = np.argsort(avg_path)            # 升序：正常→异常
avg_path_sorted = avg_path[sort_idx]

# 2. 直方图
plt.figure(figsize=(6, 4), dpi=300)
#它是 clf.decision_function(X) 返回的 负平均路径长度（乘以 -1）→ 数值 越大 表示越容易被孤立，越像异常。
sns.histplot(avg_path_sorted, kde=True, bins=30, color='steelblue')
plt.axvline(avg_path[~mask].min(), color='red', lw=2,
            label='First outlier')
plt.title('Average path length distribution')
plt.xlabel('Average path length')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'avg_path_hist.png'), dpi=300)
plt.close()

# 3. 阶梯图（按排序后索引）
plt.figure(figsize=(10, 4), dpi=300)
plt.step(range(len(avg_path_sorted)), avg_path_sorted,
         where='post', color='steelblue', lw=1)
plt.axhline(avg_path[~mask].min(), color='red', lw=2, ls='--',
            label='First outlier')
plt.title('Sorted average path length')
plt.xlabel('Sorted sample index')
plt.ylabel('Average path length')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'avg_path_step.png'), dpi=300)
plt.close()

print(f'Average-path plots saved → {out_dir}')
