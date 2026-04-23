import os

import pandas as pd
import numpy as np

from functools import reduce

# ==========================================
# 路径配置 (Path Configuration)
# 请将以下变量修改为你的实际路径
# 原始数据路径: D:\OneDrive\...\peter pan txt\二轮大修\data
# ==========================================
DATA_DIR   = ""  # 数据目录，示例：r"D:\path\to\data"
OUTPUT_DIR = ""  # 输出目录，示例：r"D:\path\to\output"；留空则输出到当前目录

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# ==========================================
# 1. 配置区域 (Configuration)
#    参考分类脚本 Peterpan_SVM_NB_simpleRegression.py
# ==========================================

# 模式选择: 'full' (所有特征) 或 'selected' (自定义特征列表)
FEATURE_MODE = 'selected'  # 可选 'full' 或 'selected'

# 如果选择 'selected'，在这里填入筛选后的特征名列表
SELECTED_FEATURES_LIST = [
    'word_1gram_但是',
    'word_1gram_一般',
    'word_2gram_但是 他们',
    'ratio_auxde3',
    'word_1gram_要是',
    'word_1gram_如果',
    'word_1gram_所以',
    'pos_3gram_wp nt wp',
    'ratio_condiConj',
    'word_1gram_男孩',
    'ratio_hypoConj',
    'word_1gram_看到',
    'word_1gram_马上',
    'ratio_er_suffix',
    'Rhyme Density',
    'word_1gram_船长',
    'word_1gram_那里',
    'word_1gram_那么',
    'word_2gram_可以 看见',
    'word_2gram_但是 没有',
    'ratio_StrongYuqi',
    'word_1gram_妈妈',
    'pos_2gram_nt wp',
    'word_1gram_所有',
    'pos_3gram_nl wp nh',
    'pos_2gram_nh nl',
    'pos_3gram_nh nl wp',
    'word_2gram_双胞胎 兄弟',
    'ratio_idiom',
    'word_1gram_太太',
    'word_1gram_下来',
    'ratio_spmark',
    'ratio_auxde1',
]

# 是否对特征做标准化
SCALE_DATA = True

# KMeans 聚类簇数
N_CLUSTERS = 3


# ==========================================
# 2. 数据读取与合并 (Data Loading & Merging)
# ==========================================


def load_and_merge_data() -> pd.DataFrame:
    """读取四个特征文件并按 file_name 合并。"""

    df_dep   = pd.read_csv(os.path.join(DATA_DIR, 'DepTag_result.csv'))
    df_lex   = pd.read_csv(os.path.join(DATA_DIR, 'Lexical&Translatibility_features.csv'))
    df_ngram = pd.read_csv(os.path.join(DATA_DIR, 'N_gram_features.csv'))
    df_read  = pd.read_csv(os.path.join(DATA_DIR, 'readibility_results.csv'))

    dfs = [
        df_dep, 
        df_lex, 
        df_ngram, 
        df_read
        ]

    # 统一去除 file_name 中可能存在的空格
    for df in dfs:
        if 'file_name' in df.columns:
            df['file_name'] = df['file_name'].str.strip()

    df_final = reduce(
        lambda left, right: pd.merge(left, right, on='file_name', how='inner'), dfs
    )

    return df_final


data = load_and_merge_data()

# 从 file_name 提取作者标签，便于计算 ARI
data['label'] = data['file_name'].apply(lambda x: ''.join([part for part in x.split('_')[0] if not part.isdigit()]))

print(f"Total samples: {len(data)}")
print(f"Total features (raw): {data.shape[1] - 2}")  # 排除 file_name 和 label
print(f"Classes found: {data['label'].unique()}")


# ==========================================
# 3. 特征选择与标准化 (Feature Selection & Scaling)
# ==========================================

non_feature_cols = ['file_name', 'label']
all_features = [c for c in data.columns if c not in non_feature_cols]

if FEATURE_MODE == 'selected':
    valid_features = [f for f in SELECTED_FEATURES_LIST if f in all_features]
    print(f"\n[Mode: Selected] Using {len(valid_features)} features.")
    if len(valid_features) < len(SELECTED_FEATURES_LIST):
        print("Warning: Some selected features were not found in the CSVs.")
    X = data[valid_features]
else:
    print(f"\n[Mode: Full] Using all {len(all_features)} features.")
    X = data[all_features]

X = X.fillna(0)

if SCALE_DATA:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X.values

y_true = data['label'].values


# ==========================================
# 4. KMeans 聚类与 ARI 评估
# ==========================================

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

ari = adjusted_rand_score(y_true, cluster_labels)
print(f"Adjusted Rand Index (KMeans, k={N_CLUSTERS}): {ari:.4f}")


# ==========================================
# 5. 使用 PCA + Matplotlib 进行可视化（参考示例图，带置信椭圆）
# ==========================================

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)

print("Explained variance ratio (PC1, PC2):", pca.explained_variance_ratio_)

# 绘图：散点 + 文本标签 + 每个簇的置信椭圆
fig, ax = plt.subplots(figsize=(12, 6))

cluster_ids = np.unique(cluster_labels)
colors = plt.cm.tab10.colors
color_map = {cid: colors[i % len(colors)] for i, cid in enumerate(cluster_ids)}

# 调整散点形状和大小，使每个类的散点有所区分
unique_sizes = np.linspace(20, 20, 20)  # 定义不同大小范围
size_map = {cid: unique_sizes[i] for i, cid in enumerate(cluster_ids)}
markers = ['o', 's', 'D']  # 圆点、正方形、菱形
marker_map = {cid: markers[i % len(markers)] for i, cid in enumerate(cluster_ids)}

for cid in cluster_ids:
    mask = cluster_labels == cid
    xs = X_2d[mask, 0]
    ys = X_2d[mask, 1]

    # 散点
    ax.scatter(xs, ys, s=size_map[cid], color=color_map[cid], label=f"{cid}", marker=marker_map[cid])

    # 文本标签（去掉年份，只保留英文，呈现30% 的标签）
    for x, y, fname in zip(xs, ys, data.loc[mask, 'file_name']):
        label = ''.join([part for part in fname.split('_')[0] if not part.isdigit()])
        if np.random.rand() <= 0.5:  # 30% 概率显示标签
            ax.text(x, y + 0.02, label, fontsize=7, ha='center', va='bottom')


    

    # 画该簇的置信椭圆
    if np.sum(mask) > 2:
        points = np.column_stack([xs, ys])
        cov = np.cov(points, rowvar=False)
        mean = points.mean(axis=0)

        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]

        # 95% 置信椭圆，对应卡方分布的临界值
        chi2_val = 5.991
        width, height = 2 * np.sqrt(vals * chi2_val)
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        ellipse = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            edgecolor=color_map[cid],
            facecolor='none',
            linestyle='--',
            linewidth=2,
            label=f'Cluster {cid} Confidence Ellipse',
        )
        ax.add_patch(ellipse)

# 左上角显示 ARI
ax.text(
    0.01,
    0.99,
    f"ARI: {ari:.3f}",
    transform=ax.transAxes,
    ha='left',
    va='top',
    fontsize=14,
)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title(f'KMeans Clustering with Confidence Ellipses (k={N_CLUSTERS})')
ax.legend(title='Cluster Label', loc='upper right')  # 修改 legend 位置到右上角
plt.tight_layout()

# 保存图片到指定路径
output_path = os.path.join(OUTPUT_DIR, 'kmeans_clustering.png')

# 修改说明：
# 1. dpi=300: 设置分辨率为 300 DPI (高清)。如果需要更高清，可设为 600。
# 2. bbox_inches='tight': 自动裁剪多余的空白边缘，防止标签被切断。
plt.savefig(output_path, dpi=600, bbox_inches='tight')

plt.show()


