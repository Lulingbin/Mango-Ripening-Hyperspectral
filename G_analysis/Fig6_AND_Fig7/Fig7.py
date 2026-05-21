# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# --------------------------------------------------
# 1. 环境配置 (保持不变)
# --------------------------------------------------
plt.rcParams['figure.facecolor'] = 'white'
sns.set(font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# --------------------------------------------------
# 2. 读数据与预处理 (保持不变)
# --------------------------------------------------
file_path = r"F:\study\manggo_program\mango_predict_all\pipeline\G_analysis\Fig6_AND_Fig7\rmse表格_noyolo .xlsx"
df = pd.read_excel(file_path)

BANDS_ORDER = ["400-1700nm", "400-1000nm", "1000-1700nm"]
MODEL_ORDER = ['Ridge', 'Lasso', 'MLP', 'SLP', 'PLSR', 'RF', 'XGB']
FEATURE_ORDER = ["all", "union", "PCA4FS", "RF4FS", "SLP4FS", "GA4FS"]

df['Bands'] = pd.Categorical(df['Bands'], categories=BANDS_ORDER, ordered=True)
df['Model'] = pd.Categorical(df['Model'], categories=MODEL_ORDER, ordered=True)
df['FeatureSet'] = pd.Categorical(df['FeatureSet'], categories=FEATURE_ORDER, ordered=True)

qing_colors = ["#BBC7BE", "#929EAB", "#84ADDC", "#7FBDDA", "#6AD1A3", "#FFD47D", "#FFA288"]
custom_cmap = LinearSegmentedColormap.from_list("QingHoliday", qing_colors, N=256)

# --------------------------------------------------
# 3. 绘图逻辑
# --------------------------------------------------
out_dir = Path('Fig6_AND_Fig7')
out_dir.mkdir(exist_ok=True)

obj_order = df['Object'].unique()
fig, axes = plt.subplots(4, 3, figsize=(15, 16), sharex=True, sharey=True)

# 强制调整子图间距，为右侧和底部留出固定空间
plt.subplots_adjust(left=0.1, right=0.85, top=0.92, bottom=0.1, wspace=0.05, hspace=0.05)

# --- 关键：先绘制所有热力图 ---
for i, obj in enumerate(obj_order):
    row_data = df[df['Object'] == obj]['RMSE']
    row_vmin, row_vmax = row_data.min(), row_data.max()

    for j, band in enumerate(BANDS_ORDER):
        ax = axes[i, j]
        table = (df.query("Object == @obj and Bands == @band")
                 .pivot_table(index='Model', columns='FeatureSet',
                              values='RMSE', aggfunc='mean', observed=True)
                 .reindex(index=MODEL_ORDER, columns=FEATURE_ORDER))

        plot_data = table.clip(lower=row_vmin)
        annot_data = table.applymap(lambda x: f'{x:.2f}' if pd.notna(x) else '')

        sns.heatmap(
            plot_data, ax=ax, annot=annot_data, fmt='',
            cmap=custom_cmap, vmin=row_vmin, vmax=row_vmax,
            cbar=False, linewidths=.5,
            annot_kws={'color': '#000000', 'fontweight': 'bold', 'size': 13}
        )

        # 修改 1：强制 X 轴标签垂直显示 (rotation=90)
        ax.set_xticklabels(FEATURE_ORDER, rotation=90, fontweight='bold', fontsize=15)
        ax.set_yticklabels(MODEL_ORDER, rotation=0, fontweight='bold', fontsize=15)

        ax.tick_params(axis='both', labelsize=14)
        for lbl in ax.get_yticklabels():
            lbl.set_fontweight('bold')

        if i == 0: ax.set_title(band, fontsize=16, fontweight='bold', pad=15)
        if j == 0:
            ax.set_ylabel(obj, fontsize=16, rotation=90, va='center', fontweight='bold', labelpad=20)
        else:
            ax.set_ylabel('')
        ax.set_xlabel('')

# --- 关键：在所有子图画完后，统一添加 Colorbar ---
# 必须先绘制一次 canvas 才能获取准确的 position
fig.canvas.draw()

for i, obj in enumerate(obj_order):
    row_data = df[df['Object'] == obj]['RMSE']
    row_vmin, row_vmax = row_data.min(), row_data.max()

    # 获取该行最后一个子图的实时位置
    last_ax_pos = axes[i, 2].get_position()

    # 修改 2：计算居中尺寸
    cbar_height_ratio = 0.8  # 颜色条高度为子图的 80%
    new_h = last_ax_pos.height * cbar_height_ratio
    offset_y = (last_ax_pos.height - new_h) / 2

    # 创建 cbar 轴 [左, 下, 宽, 高]
    cbar_ax = fig.add_axes([last_ax_pos.x1 + 0.03, last_ax_pos.y0 + offset_y, 0.015, new_h])

    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=row_vmin, vmax=row_vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('RMSE', fontsize=20, fontweight='bold')

fig.suptitle('RMSE Heatmap (Row-wise Normalized)', fontsize=20, y=0.97, fontweight='bold')

# 修改 3：保存时不再使用 bbox_inches='tight'，防止重新布局导致对不齐
f_name = out_dir / 'Fig7.png'
plt.savefig(f_name, dpi=300)
plt.show()

print('Saved successfully →', f_name.resolve())