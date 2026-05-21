# -*- coding: utf-8 -*-
"""
Best-fit Hardness – 非负偏移 + 非法点过滤 + 15×15 网格
散点一图一色 | 拟合线 & 包络带统一红色  # <<< RED
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

# ----------------------------------------------------------
EXCEL_PATH = r"目标.xlsx"
SAVE_DIR   = r"TableS1_AND_FigS1"
os.makedirs(SAVE_DIR, exist_ok=True)
# ----------------------------------------------------------

# 全局 Arial
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'

df = pd.read_excel(EXCEL_PATH, usecols="B:E")
df.columns = ['Hardness', 'TSS', 'carotenoids', 'starch_content']

def smooth(x): return np.linspace(x.min(), x.max(), 300)

# 模型库
models = {
    'Linear'     : {'xt': lambda x, c=0: x - c,
                    'yt': lambda y, d=0: y - d,
                    'inv': lambda yp, d=0: yp + d},
    'Power'      : {'xt': lambda x, c=0: np.log(x - c),
                    'yt': lambda y, d=0: np.log(y - d),
                    'inv': lambda yp, d=0: np.exp(yp) + d},
    'Exponential': {'xt': lambda x, c=0: x - c,
                    'yt': lambda y, d=0: np.log(y - d),
                    'inv': lambda yp, d=0: np.exp(yp) + d},
    'Logarithmic': {'xt': lambda x, c=0: np.log(x - c),
                    'yt': lambda y, d=0: y - d,
                    'inv': lambda yp, d=0: yp + d},
    'Polynomial2': {'xt': lambda x, c=0: np.vstack([x - c, (x - c)**2]).T,
                    'yt': lambda y, d=0: y - d,
                    'inv': lambda yp, d=0: yp + d}
}

# ==========================================================
#  核心：非负偏移 + 非法点过滤 + 15×15 网格
# ==========================================================
def fit_with_offset(x_raw, y_raw, ops):
    best_r2, best_reg, best_c, best_d, best_line = -np.inf, None, 0, 0, None
    c_range = np.linspace(0, 0.30 * x_raw.max(), 15)
    d_range = np.linspace(0, 0.30 * y_raw.max(), 15)

    for c in c_range:
        for d in d_range:
            mask = (x_raw - c > 0) & (y_raw - d > 0)
            if mask.sum() < 5:
                continue
            try:
                xt = ops['xt'](x_raw[mask], c)
                yt = ops['yt'](y_raw[mask], d)
                if xt.ndim == 1:
                    xt = xt.reshape(-1, 1)
                reg = LinearRegression().fit(xt, yt)
                r2  = r2_score(yt, reg.predict(xt))
                if r2 > best_r2:
                    best_r2 = r2
                    best_reg, best_c, best_d = reg, c, d
                    x_smooth = smooth(x_raw)
                    xt_smooth = ops['xt'](x_smooth, best_c)
                    if xt_smooth.ndim == 1:
                        xt_smooth = xt_smooth.reshape(-1, 1)
                    best_line = ops['inv'](reg.predict(xt_smooth), best_d)
            except Exception:
                continue
    return best_r2, best_reg, best_c, best_d, best_line

# ==========================================================
#  散点一图一色 | 拟合&包络统一红色  # <<< RED
# ==========================================================
DOT_COLORS = ["#6AD1A3", "#7FBDDA", "#FFA288"]   # 散点外框色
RED_LINE   = 'firebrick'                         # 拟合线+包络带统一红色

x_vars = ['TSS', 'carotenoids', 'starch_content']
x_name_1 = ['Total Soluble Solids(°Brix)', 'Carotenoids(A/g)', 'Starch content(mg/g)']
best_rows = []

scale = 3
fig, axes = plt.subplots(1, 3, figsize=(11.5*scale, 3.5*scale), sharey=True)
plt.rcParams['font.size'] = 10.5*scale

for ax, xname, x_name_unit, dot_c in zip(axes, x_vars, x_name_1, DOT_COLORS):
    x_raw = df[xname].values
    y_raw = df['Hardness'].values

    # 散点外框一图一色
    ax.scatter(x_raw, y_raw, fc='none', ec=dot_c, s=30*scale, zorder=5)

    # 最优拟合
    best_r2, best_reg, best_c, best_d, best_line = -np.inf, None, 0, 0, None
    best_name = ''
    for mname, ops in models.items():
        r2, reg, c, d, line = fit_with_offset(x_raw, y_raw, ops)
        if r2 > best_r2:
            best_r2, best_name, best_reg, best_c, best_d, best_line = r2, mname, reg, c, d, line

    # 包络带 & 拟合线统一红色
    pct = 0.45
    upper = best_line * (1 + pct)
    lower = best_line * (1 - pct)
    x_smooth = smooth(x_raw)
    ax.fill_between(x_smooth, lower, upper,
                    color=RED_LINE, alpha=0.15, lw=0, zorder=2)
    ax.plot(x_smooth, best_line, lw=2.5*scale, color=RED_LINE, zorder=3)

    ax.set_xlabel(x_name_unit, fontsize=12*scale)
    ax.set_ylabel('Firmness(N)', fontsize=12*scale)

    # 公式文本
    coef = best_reg.coef_
    inter = best_reg.intercept_
    c, d = best_c, best_d

    if best_name == 'Power':
        a = np.exp(inter)
        eq = rf"$y = {a:.3f}(x{c:+.3f})^{{{coef[0]:.3f}}}{d:+.3f}$"
    elif best_name == 'Exponential':
        a = np.exp(inter)
        eq = rf"$y = {a:.3f}\,\mathrm{{e}}^{{{coef[0]:.3f}(x{c:+.3f})}}{d:+.3f}$"
    elif best_name == 'Polynomial2':
        eq = rf"$y = {inter:.3f}{coef[0]:+.3f}(x{c:+.3f}){coef[1]:+.3f}(x{c:+.3f})^2{d:+.3f}$"
    elif best_name == 'Logarithmic':
        eq = rf"$y = {inter:.3f}{coef[0]:+.3f}\ln(x{c:+.3f}){d:+.3f}$"
    else:  # Linear
        eq = rf"$y = {inter:.3f}{coef[0]:+.3f}(x{c:+.3f}){d:+.3f}$"

    textstr = f"{best_name}   $R^2 = {best_r2:.3f}$\n{eq}"
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
            fontsize=10*scale, fontname='Arial',
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=1))

    best_rows.append({
        'X_variable': xname,
        'Best_model': best_name,
        'R2'        : best_r2,
        'c'         : best_c,
        'd'         : best_d,
        'Equation'  : eq.replace('$', '')
    })

plt.tight_layout()
out_png = os.path.join(SAVE_DIR, 'Best_fit_Hardness_Arial_yfx_offset_redLine.png')
fig.savefig(out_png, dpi=300)
fig.savefig(out_png.replace('.png', '.pdf'))
plt.show()

# 保存汇总表
csv_path = os.path.join(SAVE_DIR, 'Best_fit_summary_yfx_offset_redLine.csv')
pd.DataFrame(best_rows).to_csv(csv_path, index=False, float_format='%.4f')
print('>>> 红线版已完成！结果已保存至：', SAVE_DIR)