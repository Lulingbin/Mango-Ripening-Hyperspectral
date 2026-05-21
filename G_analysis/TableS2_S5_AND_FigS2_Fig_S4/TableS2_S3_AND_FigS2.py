from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon, binomtest


def get_writable_output_path(path: Path) -> Path:
    if not path.exists():
        return path

    try:
        with path.open("ab"):
            return path
    except PermissionError:
        return path.with_name(f"{path.stem}_new{path.suffix}")

# =========================
# 1. Load data
# =========================
base_dir = Path(__file__).resolve().parent
results_candidates = [
    base_dir / "Results_of_fittings.csv",
    base_dir / "Results_of_fittings.xlsx",
]
gt_path = base_dir / "four objectives.xlsx"

results_path = next((path for path in results_candidates if path.exists()), None)
if results_path is None:
    raise FileNotFoundError(
        "Could not find Results_of_fittings.csv or Results_of_fittings.xlsx "
        f"in {base_dir}"
    )

if results_path.suffix.lower() == ".csv":
    df = pd.read_csv(results_path)
else:
    df = pd.read_excel(results_path)

gt = pd.read_excel(gt_path)

# =========================
# 2. Standardise labels
# =========================
df["FeatureSet"] = df["FeatureSet"].replace({
    "all": "Full",
    "union": "Union"
})

df["Bands"] = df["Bands"].replace({
    "400-1700nm": "Full",
    "400-1000nm": "VNIR",
    "1000-1700nm": "SWIR"
})

df["Object"] = df["Object"].replace({
    "carotenoids": "Carotenoids",
    "firmness": "Firmness",
    "starch_content": "Starch",
    "TSS": "TSS",
    "tss": "TSS"
})

# =========================
# 3. Ground-truth statistics for NRMSE
#    NRMSE = RMSE / mean(target)
# =========================
target_stats = pd.DataFrame({
    "Object": ["Carotenoids", "Firmness", "Starch", "TSS"],
    "Target_mean": [
        gt["carotenoids"].mean(),
        gt["Hardness"].mean(),
        gt["starch content"].mean(),
        gt["TSS"].mean()
    ],
    "Target_min": [
        gt["carotenoids"].min(),
        gt["Hardness"].min(),
        gt["starch content"].min(),
        gt["TSS"].min()
    ],
    "Target_max": [
        gt["carotenoids"].max(),
        gt["Hardness"].max(),
        gt["starch content"].max(),
        gt["TSS"].max()
    ]
})

target_stats["Target_range"] = (
    target_stats["Target_max"] - target_stats["Target_min"]
)

df = df.merge(target_stats[["Object", "Target_mean"]], on="Object", how="left")

# Mean-normalised RMSE
df["NRMSE"] = df["RMSE"] / df["Target_mean"]

# =========================
# 4. Pivot to matched configurations
#    matched configuration = Bands × Object × Model
# =========================
pivot = df.pivot_table(
    index=["Bands", "Object", "Model"],
    columns="FeatureSet",
    values=["R2", "RMSE", "NRMSE"]
)

pivot.columns = [f"{metric}_{feature}" for metric, feature in pivot.columns]
pivot = pivot.reset_index()

# =========================
# 5. Define feature-selection methods
# =========================
fs_methods = ["Union", "PCA4FS", "RF4FS", "SLP4FS", "GA4FS"]

# =========================
# 6. Calculate FS-mean and Best-FS
# =========================
for metric in ["R2", "RMSE", "NRMSE"]:
    fs_cols = [f"{metric}_{m}" for m in fs_methods]

    if metric == "R2":
        # Higher is better
        pivot[f"{metric}_FS_mean"] = pivot[fs_cols].mean(axis=1)
        pivot[f"{metric}_Best_FS"] = pivot[fs_cols].max(axis=1)
    else:
        # Lower is better
        pivot[f"{metric}_FS_mean"] = pivot[fs_cols].mean(axis=1)
        pivot[f"{metric}_Best_FS"] = pivot[fs_cols].min(axis=1)

# ΔR2 > 0 means Full performs better than the average FS method.
# ΔNRMSE < 0 means Full performs better than the average FS method.
pivot["Delta_R2"] = pivot["R2_Full"] - pivot["R2_FS_mean"]
pivot["Delta_NRMSE"] = pivot["NRMSE_Full"] - pivot["NRMSE_FS_mean"]

# =========================
# 7. Statistical comparison function
# =========================
def paired_comparison(full, comparator, metric_type):
    """
    full: full-spectrum values
    comparator: FS-mean or Best-FS values

    metric_type:
        "R2"    -> higher is better
        "RMSE"  -> lower is better
        "NRMSE" -> lower is better
    """
    full = np.asarray(full, dtype=float)
    comparator = np.asarray(comparator, dtype=float)

    diff = full - comparator

    if metric_type == "R2":
        wins = diff > 0
    elif metric_type in ["RMSE", "NRMSE"]:
        wins = diff < 0
    else:
        raise ValueError("metric_type must be 'R2', 'RMSE', or 'NRMSE'.")

    # Paired t-test
    t_stat, t_p = ttest_rel(full, comparator)

    # Wilcoxon signed-rank test
    w_stat, w_p = wilcoxon(full, comparator)

    # Binomial test for win rate > 50%
    binom = binomtest(
        k=int(wins.sum()),
        n=len(wins),
        p=0.5,
        alternative="greater"
    )

    return {
        "n_configurations": len(wins),
        "Mean_Full": np.mean(full),
        "Mean_Comparator": np.mean(comparator),
        "Mean_Difference_Full_minus_Comparator": np.mean(diff),
        "Median_Difference": np.median(diff),
        "Win_Count": int(wins.sum()),
        "Win_Rate": wins.mean(),
        "Paired_t_stat": t_stat,
        "Paired_t_p": t_p,
        "Wilcoxon_stat": w_stat,
        "Wilcoxon_p": w_p,
        "Binomial_p": binom.pvalue
    }

# =========================
# 8. Run comparisons
# =========================
comparisons = []

for metric in ["R2", "RMSE", "NRMSE"]:
    comparisons.append({
        "Comparison": "Full vs FS-mean",
        "Metric": metric,
        **paired_comparison(
            pivot[f"{metric}_Full"],
            pivot[f"{metric}_FS_mean"],
            metric
        )
    })

    comparisons.append({
        "Comparison": "Full vs Best-FS",
        "Metric": metric,
        **paired_comparison(
            pivot[f"{metric}_Full"],
            pivot[f"{metric}_Best_FS"],
            metric
        )
    })

summary = pd.DataFrame(comparisons)

# =========================
# 9. Optional formatting
# =========================
summary["Win_Rate_percent"] = summary["Win_Rate"] * 100

summary = summary[
    [
        "Comparison",
        "Metric",
        "n_configurations",
        "Mean_Full",
        "Mean_Comparator",
        "Mean_Difference_Full_minus_Comparator",
        "Median_Difference",
        "Win_Count",
        "Win_Rate",
        "Win_Rate_percent",
        "Paired_t_stat",
        "Paired_t_p",
        "Wilcoxon_stat",
        "Wilcoxon_p",
        "Binomial_p"
    ]
]

# =========================
# 10. Save outputs
# =========================
output_file = "full_vs_feature_selection_R2_RMSE_NRMSE_statistics.xlsx"
output_path = get_writable_output_path(base_dir / output_file)

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    target_stats.to_excel(writer, sheet_name="target_stats", index=False)
    pivot.to_excel(writer, sheet_name="matched_configurations", index=False)
    summary.to_excel(writer, sheet_name="statistical_summary", index=False)

print(summary)
print(f"\nSaved to: {output_path.name}")


### By objective (task)-dependent FS effect analysis is appended.
# =========================
# NEW PART: Task-dependent FS effect
# =========================

# Delta_R2 has been calculated above from matched configurations.
# pivot: Bands × Object × Model

# -------------------------
# 1. 按 target 汇总
# -------------------------
target_summary = (
    pivot.groupby("Object")
    .agg(
        mean_Delta_R2=("Delta_R2", "mean"),
        median_Delta_R2=("Delta_R2", "median"),
        std_Delta_R2=("Delta_R2", "std"),
        win_rate_R2=("Delta_R2", lambda x: (x > 0).mean()),
        mean_Delta_NRMSE=("Delta_NRMSE", "mean"),
        median_Delta_NRMSE=("Delta_NRMSE", "median"),
        std_Delta_NRMSE=("Delta_NRMSE", "std"),
        win_rate_NRMSE=("Delta_NRMSE", lambda x: (x < 0).mean()),
        n=("Delta_R2", "count")
    )
    .reset_index()
    .sort_values("mean_Delta_R2", ascending=False)
)

print("\n=== Target-level sensitivity summary ===")
print(target_summary)


# -------------------------
# 2. Kruskal–Wallis test（关键）
# -------------------------
from scipy.stats import kruskal

groups_target = [
    group["Delta_R2"].values
    for _, group in pivot.groupby("Object")
]

kruskal_target = kruskal(*groups_target)

print("\n=== Kruskal–Wallis (target-level) ===")
print(f"H = {kruskal_target.statistic:.4f}, p = {kruskal_target.pvalue:.4e}")


# -------------------------
# 3. Dunn post-hoc（推荐）
# -------------------------
import scikit_posthocs as sp

dunn_target = sp.posthoc_dunn(
    pivot,
    val_col="Delta_R2",
    group_col="Object",
    p_adjust="bonferroni"
)

print("\n=== Dunn post-hoc (target-level) ===")
print(dunn_target)


# -------------------------
# 4. 保存结果（Table 可直接用）
# -------------------------
target_output_path = get_writable_output_path(base_dir / "target_dependent_FS_analysis.xlsx")

with pd.ExcelWriter(target_output_path, engine="openpyxl") as writer:
    target_summary.to_excel(writer, sheet_name="summary", index=False)
    dunn_target.to_excel(writer, sheet_name="dunn_pvalues")

print(f"\nSaved: {target_output_path.name}")

import matplotlib
import seaborn as sns

matplotlib.use("Agg")
from matplotlib import pyplot as plt

sns.set(style="whitegrid")

plt.figure(figsize=(7,5))

# 固定顺序
target_order = ["Firmness", "TSS", "Starch", "Carotenoids"]
palette = ["#fc8d62", "#66c2a5", "#8da0cb", "#ffd92f"]

# 修复：适配旧版 seaborn，无警告
sns.boxplot(
    data=pivot,
    x="Object",
    y="Delta_R2",
    order=target_order,
    hue="Object",
    palette=palette,
    width=0.25,
    fliersize=0,
    legend=False
)

# 修复：手动偏移散点，兼容所有版本
ax = plt.gca()
sns.stripplot(
    data=pivot,
    x="Object",
    y="Delta_R2",
    order=target_order,
    color="black",
    alpha=0.7,
    jitter=0.05,
    size=4,
    ax=ax
)

# 手动把散点向右偏移（兼容旧版）
for collection in ax.collections:
    if len(collection.get_offsets()) > 10:  # 找到散点
        offsets = collection.get_offsets()
        offsets[:, 0] += 0.35
        collection.set_offsets(offsets)

plt.axhline(0, linestyle="--", color="gray")
plt.xlabel("Target", fontsize=11)
plt.ylabel("ΔR² (Full − FS mean)", fontsize=11)
plt.tight_layout()
plt.savefig(base_dir / "Boxplot_Target_FS.png", dpi=300, bbox_inches='tight')
plt.close()