import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu
from matplotlib.patches import Patch

# =========================
# 1. 数据加载与预处理
# =========================
base_dir = Path(__file__).resolve().parent
results_candidates = [base_dir / "Results_of_fittings.csv", base_dir / "Results_of_fittings.xlsx"]
results_path = next((path for path in results_candidates if path.exists()), None)
gt_path = base_dir / "four objectives.xlsx"

if results_path is None: raise FileNotFoundError("Data file not found.")

df = pd.read_csv(results_path) if results_path.suffix == ".csv" else pd.read_excel(results_path)
gt = pd.read_excel(gt_path)

# 标准化标签
df["FeatureSet"] = df["FeatureSet"].replace({"all": "AllWavelengths", "union": "Union"})
df["Object"] = df["Object"].replace(
    {"carotenoids": "Carotenoids", "firmness": "Firmness", "starch_content": "Starch", "TSS": "TSS", "tss": "TSS"})

target_stats = pd.DataFrame({
    "Object": ["Carotenoids", "Firmness", "Starch", "TSS"],
    "Target_mean": [gt["carotenoids"].mean(), gt["Hardness"].mean(), gt["starch content"].mean(), gt["TSS"].mean()]
})
df = df.merge(target_stats, on="Object", how="left")
df["NRMSE"] = df["RMSE"] / df["Target_mean"]

# =========================
# 2. 计算模型敏感度 (ΔR²) 与 统计汇总
# =========================
pivot = df.pivot_table(index=["Bands", "Object", "Model"], columns="FeatureSet", values=["R2", "NRMSE"])
pivot.columns = [f"{metric}_{feature}" for metric, feature in pivot.columns]
pivot = pivot.reset_index()

fs_methods = ["Union", "PCA4FS", "RF4FS", "SLP4FS", "GA4FS"]
pivot["R2_FS_mean"] = pivot[[f"R2_{m}" for m in fs_methods]].mean(axis=1)
pivot["Delta_R2"] = pivot["R2_AllWavelengths"] - pivot["R2_FS_mean"]

# 计算 ΔNRMSE 用于 Excel 备份数据完整性
pivot["Delta_NRMSE"] = pivot["NRMSE_AllWavelengths"] - pivot[[f"NRMSE_{m}" for m in fs_methods]].mean(axis=1)

model_groups = {"Ridge": "Linear", "Lasso": "Linear", "MLP": "NN", "SLP": "NN", "PLSR": "Linear", "RF": "Tree", "XGB": "Tree"}
pivot["Group"] = pivot["Model"].map(model_groups)

# 生成模型级指标汇总表（用于写入 Excel）
summary = (
    pivot.groupby("Model")
    .agg(
        mean_Delta_R2=("Delta_R2", "mean"),
        median_Delta_R2=("Delta_R2", "median"),
        sd_Delta_R2=("Delta_R2", "std"),
        mean_Delta_NRMSE=("Delta_NRMSE", "mean"),
        median_Delta_NRMSE=("Delta_NRMSE", "median"),
        sd_Delta_NRMSE=("Delta_NRMSE", "std"),
        n=("Delta_R2", "count")
    )
    .reset_index()
    .sort_values("mean_Delta_R2", ascending=False)
)

# =========================
# 3. 绘图参数设置 (实现箱点分离)
# =========================
sns.set_theme(style="ticks")
plot_order = ["Ridge", "Lasso", "MLP", "SLP", "PLSR", "XGB", "RF"]
plot_data = pivot[pivot["Model"].isin(plot_order)].copy()
palette = {"Linear": "#fc8d62", "NN": "#66c2a5", "Tree": "#8da0cb"}

fig, ax = plt.subplots(figsize=(10, 6))

box_width = 0.25   # 缩小箱子宽度
dot_offset = 0.20  # 点向右偏移的距离
x_base = np.arange(len(plot_order))

# 循环绘制每一组，确保箱和点分开
for i, model in enumerate(plot_order):
    m_data = plot_data[plot_data["Model"] == model]
    m_group = m_data["Group"].iloc[0]

    # 绘制箱线图 (左侧)
    ax.boxplot(
        m_data["Delta_R2"],
        positions=[i],
        widths=box_width,
        patch_artist=True,
        boxprops=dict(facecolor=palette[m_group], alpha=0.8, linewidth=1.1),
        medianprops=dict(color='black', linewidth=1.1),
        showfliers=False,
        zorder=2
    )

    # 绘制散点图 (右侧偏移)
    x_val = np.full(len(m_data), i + dot_offset) + np.random.uniform(-0.03, 0.03, len(m_data))
    ax.scatter(x_val, m_data["Delta_R2"], color=".3", alpha=0.4, s=12, zorder=1)


# =========================
# 4. 针对性显著性标注
# =========================
def get_star(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    return "n.s."

# 仅对比指定组
specific_comparisons = [("Ridge", "RF"), ("Ridge", "XGB"), ("Lasso", "RF"), ("Lasso", "XGB")]
y_max = plot_data["Delta_R2"].max()
y_start = y_max + 0.03
line_gap = 0.04

for idx, (m1, m2) in enumerate(specific_comparisons):
    d1 = plot_data[plot_data.Model == m1].Delta_R2
    d2 = plot_data[plot_data.Model == m2].Delta_R2

    stat, p = mannwhitneyu(d1, d2, alternative='two-sided')
    star = get_star(p)

    x1, x2 = plot_order.index(m1), plot_order.index(m2)
    y = y_start + idx * line_gap

    # 绘制横线和标注
    ax.plot([x1, x1, x2, x2], [y, y + 0.005, y + 0.005, y], lw=1, c='0.3')
    ax.text((x1 + x2) / 2, y + 0.005, star, ha='center', va='bottom', fontsize=10, fontweight='bold')

# =========================
# 5. 细节美化
# =========================
ax.axhline(0, linestyle="--", color="gray", linewidth=1, zorder=0)
ax.set_xticks(x_base)
ax.set_xticklabels(plot_order, fontsize=11)
ax.set_ylabel("$\Delta R^2$ (All-wavelengths $-$ FS)", fontsize=12)
ax.set_xlabel("Regression Model", fontsize=12)
ax.set_ylim(plot_data["Delta_R2"].min() - 0.05, y_start + len(specific_comparisons) * line_gap + 0.05)

# 制作图例
legend_elements = [Patch(facecolor=palette[k], label=k) for k in palette]
ax.legend(handles=legend_elements, title="Model Category", loc='upper right')

sns.despine()
plt.title("Model Sensitivity to Feature Selection (Specific Comparisons)", fontsize=14, pad=20)
plt.tight_layout()

# =========================
# 6. 安全的数据存盘机制 (融合代码一逻辑)
# =========================
output_file = "model_dependent_feature_selection_sensitivity.xlsx"
output_path = base_dir / output_file

# 安全性检查：如果原 Excel 被打开占用了，自动存入新文件中防止代码崩溃
if output_path.exists():
    try:
        with output_path.open("ab"): pass
    except PermissionError:
        output_path = output_path.with_name(f"{output_path.stem}_new{output_path.suffix}")

# 写入 Excel 多个工作表
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    pivot.to_excel(writer, sheet_name="matched_configurations", index=False)
    summary.to_excel(writer, sheet_name="model_summary", index=False)

print(f"Excel 统计数据已成功保存至: {output_path.name}")

# 保存精美的高清图片
plt.savefig(base_dir / "Figure_Model_Sensitivity_Clean.png", dpi=300)
print(f"精美统计图已成功保存至: Figure_Model_Sensitivity_Clean.png")

plt.show()