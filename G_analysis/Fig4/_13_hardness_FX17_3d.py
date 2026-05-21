def save_13bands(csv_path, png_path):
    # -*- coding: utf-8 -*-
    """
    FX17  13 硬度区间 3D 光谱曲面图（112 波段）
    样品按硬度排序，只生成指定视角视图 - 优化版（更大字体、更宽敞布局）
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    from matplotlib import rcParams
    from mpl_toolkits.mplot3d import Axes3D
    import os

    # 创建输出目录
    output_dir = 'FX17_3D_Results2'
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------
    # 0. 波长映射 225-336 (FX17波段)
    # ----------------------------------------------------------
    wave_df = pd.read_excel(
        r'F:\study\manggo_program\Easypilpeline\new1.0\G_analysis\13_hardness\bands_and_names.xlsx'
    )
    wave_df = wave_df[wave_df['id'] > 224].sort_values('id')
    wavelengths = wave_df['Wavelength (nm)'].values  # 112

    # ----------------------------------------------------------
    # 1. 加载 FX17 数据（112 波段）+ 硬度值
    # ----------------------------------------------------------
    fx17 = pd.read_csv(csv_path)

    # 按硬度排序！！！
    fx17 = fx17.sort_values('Hardness').reset_index(drop=True)
    print(f"样品已按硬度排序: {fx17['Hardness'].min():.1f} -> {fx17['Hardness'].max():.1f}")

    # 获取硬度值用于着色
    hardness_values = fx17['Hardness'].values

    # 提取 112 波段光谱数据 (Band_1 到 Band_112 对应 FX17)
    band_cols = [f'Band_{i}' for i in range(1, 113)]
    spectra = fx17[band_cols].values  # shape: (n_samples, 112)

    n_samples = spectra.shape[0]
    n_bands = spectra.shape[1]

    print(f"样品数: {n_samples}, 波段数: {n_bands}")
    print(f"硬度范围: {hardness_values.min():.1f} - {hardness_values.max():.1f}")

    # ----------------------------------------------------------
    # 2. 科研风设置 - 【大幅增大字体和画布】
    # ----------------------------------------------------------
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']

    # 增大缩放因子
    n = 3  # 从2改为3，字体更大

    # ========== 全局字体大小设置（大幅增大）==========
    rcParams['font.size'] = 20 * n  # 基础字体从16改为20
    rcParams['axes.titlesize'] = 22 * n  # 坐标轴标题从18改为22
    rcParams['axes.labelsize'] = 20 * n  # 坐标轴标签从16改为20
    rcParams['xtick.labelsize'] = 18 * n  # X轴刻度标签从14改为18
    rcParams['ytick.labelsize'] = 18 * n  # Y轴刻度标签从14改为18
    rcParams['xtick.major.size'] = 8 * n  # 刻度线长度从6改为8
    rcParams['ytick.major.size'] = 8 * n
    rcParams['axes.linewidth'] = 2.0  # 坐标轴线宽从1.2改为2.0

    # ----------------------------------------------------------
    # 3. 创建颜色映射（基于硬度值）
    # ----------------------------------------------------------
    norm = mcolors.Normalize(vmin=10, vmax=140)
    cmap = cm.get_cmap('turbo')
    sample_colors = cmap(norm(hardness_values))

    # ----------------------------------------------------------
    # 4. 3D 绘图 - 【大幅增大画布尺寸】
    # ----------------------------------------------------------
    print("\n正在生成 3D 线框图...")

    # 大幅增大画布尺寸，给字体留出充足空间
    fig = plt.figure(figsize=(24, 18))  # 从(18, 12)改为(24, 18)
    ax = fig.add_subplot(111, projection='3d')

    # 每隔N个样品绘制一条线（根据FX17样品数调整）
    step = max(1, n_samples // 633)

    for i in range(0, n_samples, step):
        color = sample_colors[i]
        ax.plot(wavelengths, [i] * n_bands, spectra[i],
                color=color, alpha=0.7, linewidth=1.2)  # 线宽稍微增加

    # ========== 颜色条字体大小（大幅增大）==========
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    # 调整颜色条位置和大小，避免拥挤
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=15, pad=0.15)  # shrink从0.6改为0.5，pad从0.12改为0.15
    cbar.set_label('Firmness (N)', fontsize=22 * n, fontweight='bold', labelpad=20)  # 从18改为22，增加labelpad
    cbar.ax.tick_params(labelsize=18 * n)  # 从14改为18

    # 颜色条刻度线加粗
    cbar.ax.tick_params(axis='y', which='major', labelsize=18 * n, width=2.5, length=8)

    # ========== 坐标轴标签字体大小（大幅增大）==========
    ax.set_xlabel('Wavelength (nm)', fontsize=22 * n, labelpad=55, fontweight='bold')  # 18→22, pad 20→25
    ax.set_ylabel('', fontsize=20 * n, labelpad=20)  # 16→20, pad 15→20
    ax.set_zlabel('Reflectance', fontsize=22 * n, labelpad=60, fontweight='bold')  # 18→22, pad 20→25

    # 设置刻度标签大小
    ax.tick_params(axis='x', labelsize=18 * n, pad=12)  # 14→18, pad 10→12
    ax.tick_params(axis='y', labelsize=18 * n, pad=12)  # 14→18, pad 10→12
    ax.tick_params(axis='z', labelsize=18 * n, pad=26)  # 14→18, pad 10→12

    # 计算数据范围
    x_min, x_max = wavelengths.min(), wavelengths.max()
    y_min, y_max = 0, n_samples
    z_min, z_max = spectra.min(), spectra.max()

    # 扩大坐标轴范围，给标签留出空间
    x_margin = (x_max - x_min) * 0.12  # 从0.08改为0.12
    y_margin = (y_max - y_min) * 0.12  # 从0.08改为0.12
    z_margin = (z_max - z_min) * 0.15  # 从0.10改为0.15

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_zlim(z_min - z_margin, z_max + z_margin)

    # 设置指定视角
    ax.view_init(elev=30, azim=-115)

    # ----------------------------------------------------------
    # 5. 去除坐标轴刻度线
    # ----------------------------------------------------------
    ax.xaxis._axinfo["tick"]["inward_factor"] = 0.0
    ax.xaxis._axinfo["tick"]["outward_factor"] = 0.0
    ax.yaxis._axinfo["tick"]["inward_factor"] = 0.0
    ax.yaxis._axinfo["tick"]["outward_factor"] = 0.0
    ax.zaxis._axinfo["tick"]["inward_factor"] = 0.0
    ax.zaxis._axinfo["tick"]["outward_factor"] = 0.0

    # 调整布局 - 使用更大的边距
    plt.tight_layout(pad=3.0)  # 增加pad值

    # 保存图像 - 【增加边距和DPI】
    filename = png_path
    filepath = os.path.join(output_dir, filename)
    # pad_inches从0.4改为0.8，给边缘留出更多空间
    fig.savefig(filepath, dpi=600, bbox_inches='tight', pad_inches=0.8)
    print(f"  已保存: {filename}")

    plt.close(fig)
    print(f"\n图像已保存到: {os.path.abspath(filepath)}")


# ----------------------------------------------------------
# 运行部分保持不变
# ----------------------------------------------------------
raw_file = r'F:\study\manggo_program\Easypilpeline\new1.0\A_Dataset\FX17\Hardness_10_140\outlier_isolationforest\clean.csv'
raw_out = 'FX17_mean_spectra_13groups_trueWave_raw.png'

sg_file = r'F:\study\manggo_program\Easypilpeline\new1.0\A_Dataset\FX17\Hardness_10_140\outlier_isolationforest\Dataset\all\train_SG_all.csv'
sg_out = 'FX17_mean_spectra_13groups_trueWave_Sg.png'

snv_file = r'F:\study\manggo_program\Easypilpeline\new1.0\A_Dataset\FX17\Hardness_10_140\outlier_isolationforest\Dataset\all\train_SNV_all.csv'
snv_out = 'FX17_mean_spectra_13groups_trueWave_SNV.png'

zscore_file = r'F:\study\manggo_program\Easypilpeline\new1.0\A_Dataset\FX17\Hardness_10_140\outlier_isolationforest\Dataset\all\train_all.csv'
zscore_out = 'FX17_mean_spectra_13groups_z-score.png'

files = [raw_file, sg_file, snv_file, zscore_file]
outs = [raw_out, sg_out, snv_out, zscore_out]

for i, (f, o) in enumerate(zip(files, outs)):
    save_13bands(f, o)