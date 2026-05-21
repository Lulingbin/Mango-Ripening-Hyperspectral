# -*- coding: utf-8 -*-
"""
控制变量实验结果可视化 —— 重写版
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import colors
import numpy as np

plt.rcParams['figure.facecolor'] = 'white'
sns.set(font_scale=1.1)
# 全局改用 Arial
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# --------------------------------------------------
# 1. 读数据
# --------------------------------------------------
file_path =  r"F:\study\manggo_program\Easypilpeline\new1.0\A_Dataset\总结\表格_noyolo.xlsx"        # 换成你的文件名
df = pd.read_excel(file_path)

# --------------------------------------------------
# 2. 排序字典
# --------------------------------------------------
BANDS_ORDER   = [ "400-1700nm","400-1000nm", "1000-1700nm"]
MODEL_ORDER   = ['Ridge','Lasso','MLP','SLP','PLSR','RF','XGB',]#'YOLO1D'
FEATURE_ORDER = ["all","union","PCA4FS","RF4FS","SLP4FS","GA4FS"]

df['Bands']      = pd.Categorical(df['Bands'],      categories=BANDS_ORDER,   ordered=True)
df['Model']      = pd.Categorical(df['Model'],      categories=MODEL_ORDER,   ordered=True)
df['FeatureSet'] = pd.Categorical(df['FeatureSet'], categories=FEATURE_ORDER, ordered=True)

# --------------------------------------------------
# 3. 4×3 一张图：0 以下纯白，0–1 渐变，数字全保留
# --------------------------------------------------
out_dir = Path('Fig6_AND_Fig7')
out_dir.mkdir(exist_ok=True)

# 统一颜色范围 0–1
vmin, vmax = 0, 1
# cmap = sns.color_palette('YlOrRd', as_cmap=True)
cmap = plt.get_cmap('coolwarm')

fig, axes = plt.subplots(4, 3, figsize=(15, 18), sharex=True, sharey=True)
obj_order = df['Object'].unique()

for i, obj in enumerate(obj_order):
    for j, band in enumerate(BANDS_ORDER):
        ax = axes[i, j]

        table = (df.query("Object == @obj and Bands == @band")
                   .pivot_table(index='Model', columns='FeatureSet',
                                values='R2', aggfunc='mean', observed=True)
                   .reindex(index=MODEL_ORDER, columns=FEATURE_ORDER))

        # 清新假日连续渐变 colormap  # <<< NEW
        # ------------------------------------------------------
        # qing_colors = [
        #     "#BBC7BE", "#929EAB", "#84ADDC", "#7FBDDA",
        #     "#6AD1A3", "#FFD47D", "#C49892", "#FFA288"
        # ]
        qing_colors = [
            "#BBC7BE", "#929EAB", "#84ADDC", "#7FBDDA",
            "#6AD1A3", "#FFD47D", "#FFA288"
        ]
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("QingHoliday", qing_colors, N=256)

        # -------------- 只替换这一段即可 --------------
        # 1. 上色用：把 <0 当成 0 处理，保证颜色不溢出
        plot_data = table.clip(lower=0)

        # 2. 注释用：真实值，<0 的也照常显示
        annot_data = table.applymap(lambda x: f'{x:.3f}' if pd.notna(x) else '')

        sns.heatmap(
            plot_data,
            ax=ax,
            annot=annot_data,
            fmt='',
            cmap=cmap,
            vmin=0, vmax=1,
            cbar=False,
            linewidths=.5,
            # 关键修改：使用十六进制纯黑 + 关闭抗锯齿
            annot_kws={
                'color': '#000000',  # 强制纯黑，不用 'black' 关键词
                'fontweight': 'bold',
                'size': 15,
                'alpha': 1.0,  # 确保不透明
                'antialiased': False  # 关闭抗锯齿（可选）
            }
        )
        # 统一改 ticklabels 字号 & 加粗
        ax.tick_params(axis='both', labelsize=15)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')

        # 轴标签控制同之前
        if i == 0:
            ax.set_title(band, fontsize=15,fontweight='bold')
        else:
            ax.set_title('')
        if j == 0:
            ax.set_ylabel(obj, fontsize=15, rotation=90, va='center',fontweight='bold')
        else:
            ax.set_ylabel('')
        ax.set_xlabel('')

# 公共 colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, ticks=np.linspace(0, 1, 6))
cbar.ax.tick_params(labelsize=20)      # 刻度数字
cbar.set_label('R²', fontsize=20,fontweight='bold')

fig.suptitle('R² heatmap', fontsize=20, y=0.98)
plt.tight_layout(rect=[0, 0, 0.9, 0.96])
f_name = out_dir / 'all_in_one_white_below_0.png'
plt.savefig(f_name, dpi=300)
plt.close()
print('saved →', f_name.resolve())
