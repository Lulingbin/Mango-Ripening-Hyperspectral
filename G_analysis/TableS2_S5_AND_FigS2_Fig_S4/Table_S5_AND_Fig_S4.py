import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp
from matplotlib.patches import Patch

# =========================
# 1. 数据加载与预处理
# =========================
results_path = "Results_of_fittings.xlsx"
df = pd.read_excel(results_path)

df["Bands"] = df["Bands"].replace({"400-1700nm": "Combined", "400-1000nm": "VNIR", "1000-1700nm": "SWIR"})
df["Object"] = df["Object"].replace(
    {"carotenoids": "Carotenoids", "firmness": "Firmness", "starch_content": "Starch", "TSS": "TSS", "tss": "TSS"})

# =========================
# 2. 绘图与参数设置
# =========================
sns.set_theme(style="ticks")
band_order = ["Combined", "VNIR", "SWIR"]
obj_order = ["Carotenoids", "Firmness", "Starch", "TSS"]
palette = {"Combined": "#4C72B0", "VNIR": "#55A868", "SWIR": "#C44E52"}

fig, ax = plt.subplots(figsize=(15, 9))
box_width, group_offset, line_gap = 0.10, 0.28, 0.08
x_base = np.arange(len(obj_order))
offsets = [-group_offset, 0, group_offset]


def get_sig_label(p):
    if pd.isna(p): return "n.s."
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


# =========================
# 3. 核心逻辑：Friedman 检验 + Wilcoxon 配对检验
# =========================
all_posthoc_data = []
friedman_results = []

print("=" * 50)
print("统计分析启动：Friedman 整体检验及 Wilcoxon 事后分析")
print("=" * 50)

for j, obj in enumerate(obj_order):
    # 筛选并准备数据
    obj_data = df[df["Object"] == obj].copy()
    obj_data = obj_data.sort_values(by=['Bands'])
    obj_data['Block'] = obj_data.groupby('Bands').cumcount()

    # 转换为矩阵以确保“配对”属性
    matrix_data = obj_data.pivot(index='Block', columns='Bands', values='R2')[band_order]

    # --- 新增：Friedman Test ---
    # 检验 Combined, VNIR, SWIR 三者之间是否存在显著差异
    f_stat, f_p = friedmanchisquare(matrix_data['Combined'], matrix_data['VNIR'], matrix_data['SWIR'])

    friedman_results.append({'Object': obj, 'Friedman_Stat': f_stat, 'p_value': f_p})

    print(f"\n[指标: {obj}]")
    print(f"Friedman Test: Statistic={f_stat:.4f}, p-value={f_p:.4e}")

    if f_p < 0.05:
        print(f"-> 结果显著 (p < 0.05)，继续进行 Wilcoxon 事后配对分析...")
    else:
        print(f"-> 结果不显著 (p >= 0.05)，组间无统计学差异。")

    # A. 绘图部分 (箱线图与散点)
    for i, band in enumerate(band_order):
        band_series = obj_data[obj_data["Bands"] == band]["R2"]
        curr_pos = j + offsets[i]
        ax.boxplot(band_series, positions=[curr_pos], widths=box_width, patch_artist=True,
                   boxprops=dict(facecolor=palette[band], alpha=0.8), showfliers=False)
        x_scatter = np.full(len(band_series), curr_pos + 0.08) + np.random.uniform(-0.02, 0.02, len(band_series))
        ax.scatter(x_scatter, band_series, color=".3", alpha=0.3, s=8)

    # B. 统计计算 (Wilcoxon 事后检验)
    p_df = pd.DataFrame(np.eye(3), index=band_order, columns=band_order)
    pairs = [("Combined", "VNIR"), ("VNIR", "SWIR"), ("Combined", "SWIR")]

    for b1, b2 in pairs:
        _, p = wilcoxon(matrix_data[b1], matrix_data[b2])
        # Bonferroni 校正
        adj_p = min(p * 3, 1.0)
        p_df.loc[b1, b2] = p_df.loc[b2, b1] = adj_p

    # C. 绘制显著性标注
    # 动态计算标注起始高度
    y_start = obj_data["R2"].max() + 0.05
    for idx, (b1, b2) in enumerate(pairs):
        p_val = p_df.loc[b1, b2]
        label = get_sig_label(p_val)
        x1, x2 = j + offsets[band_order.index(b1)], j + offsets[band_order.index(b2)]
        y = y_start + idx * line_gap

        # 只有在 Friedman 显著的情况下才标注显著性（可选，此处保留全部标注）
        ax.plot([x1, x1, x2, x2], [y, y + 0.01, y + 0.01, y], lw=1, c='0.3')
        weight = 'bold' if label != "n.s." else 'normal'
        ax.text((x1 + x2) / 2, y + 0.01, label, ha='center', va='bottom', fontsize=10, fontweight=weight)

    res_to_save = p_df.copy()
    res_to_save['Object'] = obj
    all_posthoc_data.append(res_to_save)

# =========================
# 4. 导出结果与图形美化
# =========================
# 保存 Friedman 检验结果
df_friedman = pd.DataFrame(friedman_results)
df_friedman.to_excel("Friedman_Test_Results.xlsx", index=False)

ax.set_xticks(x_base)
ax.set_xticklabels(obj_order, fontsize=12)
ax.set_ylabel("Prediction Accuracy ($R^2$)", fontsize=14)
# 自动调整 Y 轴上限
ax.set_ylim(0, df["R2"].max() + 0.4)

legend_elements = [Patch(facecolor=palette[b], label=b) for b in band_order]
ax.legend(handles=legend_elements, title="Spectral Range", loc='lower right')

sns.despine()
plt.title("Statistical Comparison (Friedman Test + Wilcoxon Post-hoc)", fontsize=16, pad=35)
plt.tight_layout()
plt.savefig("Figure3_5_Full_Stats_Updated.png", dpi=300)

pd.concat(all_posthoc_data).to_excel("Statistical_Results_Final.xlsx")

print("\n" + "=" * 50)
print("所有分析已完成！")
print("1. 图表已保存至: Figure3_5_Full_Stats_Updated.png")
print("2. Friedman 结果已保存至: Friedman_Test_Results.xlsx")
print("3. 事后检验结果已保存至: Statistical_Results_Final.xlsx")
print("=" * 50)
plt.show()