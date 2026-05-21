# -*- coding: utf-8 -*-
"""
Spearman 相关性热力图 + 系数/p 值表格
Arial 字体，与主图风格统一
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr
import numpy as np

# ----------------------------------------------------------
EXCEL_PATH = r"目标.xlsx"
SAVE_DIR   = r"TableS1_AND_FigS1"
os.makedirs(SAVE_DIR, exist_ok=True)

# 全局 Arial
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
# ----------------------------------------------------------

df = pd.read_excel(EXCEL_PATH, usecols="B:E")
df.columns = ['Hardness', 'TSS', 'carotenoids', 'starch_content']

# 中文/英文双标签，热力图用英文，表格可保留中文
nice_names = {
    'Hardness'      : 'Firmness',
    'TSS'           : 'TSS',
    'carotenoids'   : 'Carotenoids',
    'starch_content': 'Starch'
}
df_renamed = df.rename(columns=nice_names)

# 计算 Spearman r 与 p
def spearman_frame(data):
    cols = data.columns
    n = len(cols)
    r_mat = np.zeros((n, n))
    p_mat = np.zeros((n, n))
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            r, p = spearmanr(data[ci], data[cj])
            r_mat[i, j] = r
            p_mat[i, j] = p
    r_df = pd.DataFrame(r_mat, index=cols, columns=cols)
    p_df = pd.DataFrame(p_mat, index=cols, columns=cols)
    return r_df, p_df

r_df, p_df = spearman_frame(df_renamed)

# ----------------------------------------------------------
# 1. 热力图（Arial、大尺寸、横坐标标签水平）
# ----------------------------------------------------------
plt.figure(figsize=(9, 5))          # 比原来更大
mask = np.triu(np.ones_like(r_df, dtype=bool))
# ax = sns.heatmap(
#         r_df, mask=mask,
#         annot=True, fmt='.3f',
#         cmap='coolwarm', vmin=-1, vmax=1,
#         square=True, linewidths=.5,
#         cbar_kws={"shrink": .8},
#         annot_kws={'fontsize': 11}   # 系数字号
# )
ax = sns.heatmap(
        r_df,
        annot=True, fmt='.3f',
        cmap='coolwarm', vmin=-1, vmax=1,
        square=True, linewidths=.5,
        cbar_kws={"shrink": .8},
        annot_kws={'fontsize': 11}   # 系数字号
)
ax.set_title('Spearman correlation coefficient', fontsize=14, pad=15)

# 横纵坐标均水平显示
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=11)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center', fontsize=11)

plt.tight_layout()
for ext in ['png', 'pdf']:
    f_out = os.path.join(SAVE_DIR, f'Spearman_heatmap.{ext}')
    plt.savefig(f_out, dpi=300)
plt.show()


# 2. 系数 + p 值 表格（长表，方便直接贴论文）
stack_r = r_df.stack().reset_index()
stack_p = p_df.stack().reset_index()
stack_r.columns = ['Variable 1', 'Variable 2', 'r']
stack_p.columns = ['Variable 1', 'Variable 2', 'p']
table = pd.concat([stack_r, stack_p['p']], axis=1)
table['Significance'] = table['p'].apply(
    lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else 'ns')

# 保存
csv_path = os.path.join(SAVE_DIR, 'Spearman_corr_p.csv')
xlsx_path = os.path.join(SAVE_DIR, 'Spearman_corr_p.xlsx')
table.to_csv(csv_path, index=False, float_format='%.4f')
table.to_excel(xlsx_path, index=False)

print('>>> Spearman 分析完成，结果已保存至：', SAVE_DIR)


# ==========================================================
# 快速正态/偏态体检（样本量大，走 Shapiro 太严，用偏度+峰度+QQ 图）
# ==========================================================
from scipy.stats import skew, kurtosis
import scipy.stats as st
import pylab

def quick_norm_check(data, var_names, save_dir):
    """返回 DataFrame：偏度、峰度、Jarque-Bera p 值；并画出 QQ 图"""
    res = []
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    n_var = len(var_names)
    fig, axes = plt.subplots(1, n_var, figsize=(3*n_var, 4))
    if n_var == 1:
        axes = [axes]
    for ax, v in zip(axes, var_names):
        x = data[v].dropna()
        s = skew(x)
        k = kurtosis(x, fisher=False)      # 正态=3
        jb, jb_p = st.jarque_bera(x)       # 原假设：样本来自正态分布
        res.append({'Variable': v,
                    'Skewness': s,
                    'Kurtosis': k,
                    'JB_p': jb_p})
        # QQ 图
        st.probplot(x, dist="norm", plot=ax)
        ax.set_title(f'{v}\nSkew={s:.2f}, Kurt={k:.2f}, JB_p={jb_p:.3f}', fontsize=10)
    plt.tight_layout()
    out = os.path.join(save_dir, 'Normality_check.png')
    plt.savefig(out, dpi=300)
    plt.show()
    return pd.DataFrame(res)

norm_df = quick_norm_check(df_renamed, df_renamed.columns, SAVE_DIR)

# ----------------------------------------------------------
# 自动判据：JB_p>0.05 且 |Skew|<1 且 |Kurt-3|<2 视为“近似正态”
# ----------------------------------------------------------
norm_df['ApproxNormal'] = (
        (norm_df['JB_p'] > 0.05) &
        (np.abs(norm_df['Skewness']) < 1.0) &
        (np.abs(norm_df['Kurtosis'] - 3) < 2.0)
)

print('\n>>> 正态性快速体检结果：')
print(norm_df)

if norm_df['ApproxNormal'].all():
    print('\n>>> 全部变量近似正态 → 可直接使用 Pearson 分析（前文已生成）')
else:
    print('\n>>> 存在明显非正态变量 → 建议优先使用 Spearman 结果，或对变量做变换后再 Pearson')


# ==========================================================
# 新增：Pearson 相关性热力图 + 系数/p 值表格
# ==========================================================
from scipy.stats import pearsonr

def pearson_frame(data):
    cols = data.columns
    n = len(cols)
    r_mat = np.zeros((n, n))
    p_mat = np.zeros((n, n))
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            r, p = pearsonr(data[ci], data[cj])
            r_mat[i, j] = r
            p_mat[i, j] = p
    r_df = pd.DataFrame(r_mat, index=cols, columns=cols)
    p_df = pd.DataFrame(p_mat, index=cols, columns=cols)
    return r_df, p_df

r_pea_df, p_pea_df = pearson_frame(df_renamed)



# ----------------------------------------------------------
# 1. Pearson 热力图
# ----------------------------------------------------------
plt.figure(figsize=(9, 5))
ax = sns.heatmap(
        r_pea_df,
        annot=True, fmt='.3f',
        cmap='coolwarm', vmin=-1, vmax=1,
        square=True, linewidths=.5,
        cbar_kws={"shrink": .8},
        annot_kws={'fontsize': 11}
)
ax.set_title('Pearson correlation coefficient', fontsize=14, pad=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=11)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center', fontsize=11)
plt.tight_layout()
for ext in ['png', 'pdf']:
    f_out = os.path.join(SAVE_DIR, f'Pearson_heatmap.{ext}')
    plt.savefig(f_out, dpi=300)
plt.show()

# ----------------------------------------------------------
# 2. Pearson 系数 + p 值 长表
# ----------------------------------------------------------
stack_r_pea = r_pea_df.stack().reset_index()
stack_p_pea = p_pea_df.stack().reset_index()
stack_r_pea.columns = ['Variable 1', 'Variable 2', 'r']
stack_p_pea.columns = ['Variable 1', 'Variable 2', 'p']
table_pea = pd.concat([stack_r_pea, stack_p_pea['p']], axis=1)
table_pea['Significance'] = table_pea['p'].apply(
    lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else 'ns')

csv_path_pea = os.path.join(SAVE_DIR, 'Pearson_corr_p.csv')
xlsx_path_pea = os.path.join(SAVE_DIR, 'Pearson_corr_p.xlsx')
table_pea.to_csv(csv_path_pea, index=False, float_format='%.4f')
table_pea.to_excel(xlsx_path_pea, index=False)

print('>>> Pearson 分析完成，结果已保存至：', SAVE_DIR)