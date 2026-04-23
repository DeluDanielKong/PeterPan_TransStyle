import os

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from functools import reduce

# ==========================================
# 路径配置 (Path Configuration)
# 请将 DATA_DIR 修改为你的数据目录路径
# 原始路径: D:\OneDrive\...\peter pan txt\二轮大修\data
# ==========================================
DATA_DIR = ""  # 示例：r"D:\path\to\data"

# ==========================================
# 1. 基础配置与文件读取
# ==========================================

files = {
    'Dependency':  os.path.join(DATA_DIR, 'DepTag_result.csv'),
    'Lexical':     os.path.join(DATA_DIR, 'Lexical&Translatibility_features.csv'),
    'N-gram':      os.path.join(DATA_DIR, 'N_gram_features.csv'),
    'Readability': os.path.join(DATA_DIR, 'readibility_results.csv'),
}

data_frames = []
feature_source_map = {} # 用于记录每个特征属于哪个层级(Category)

# --- 文件名清洗函数 ---
# 解决文件名后缀不一致的问题 (例如 _pos.txt, _pos)
def clean_filename(fname):
    fname = fname.strip()
    if fname.endswith('_pos.txt'):
        return fname.replace('_pos.txt', '')
    elif fname.endswith('_pos'):
        return fname.replace('_pos', '')
    return fname

print("正在读取并对齐文件...")

for category, filename in files.items():
    try:
        df = pd.read_csv(filename)
        
        # 清洗文件名列，作为合并的Key
        if 'file_name' in df.columns:
            df['file_name'] = df['file_name'].apply(clean_filename)
        
        # 记录特征来源 (排除 file_name)
        features = [c for c in df.columns if c != 'file_name']
        for f in features:
            feature_source_map[f] = category
            
        data_frames.append(df)
        print(f"  -> 已加载 {category} 特征: {len(features)} 个")
    except Exception as e:
        print(f"  [Error] 读取 {filename} 失败: {e}")

# ==========================================
# 2. 合并数据 (Merge)
# ==========================================

if data_frames:
    # 使用 inner join 确保只保留所有文件中都存在的样本
    df_final = reduce(lambda left, right: pd.merge(left, right, on='file_name', how='inner'), data_frames)
else:
    raise ValueError("没有加载到有效数据")

print(f"合并后数据维度: {df_final.shape}")

# ==========================================
# 3. 准备数据 (X, y)
# ==========================================

# 提取标签 (假设文件名格式为 Label_Chapter，如 Liang1929_10 -> Label: Liang1929)
df_final['label'] = df_final['file_name'].apply(lambda x: x.split('_')[0])

X = df_final.drop(columns=['file_name', 'label'])
y = df_final['label']

# --- 预处理 (Chi2 要求非负值) ---
# 1. 填充缺失值
X = X.fillna(0)

# 2. 归一化 (Scaling to 0-1)
# 如果数据中有负值（如 z-score 或 PCA结果），Chi2会报错，所以必须先MinMax
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ==========================================
# 4. Chi-Square 特征筛选
# ==========================================

print("正在计算 Chi2 贡献度...")
selector = SelectKBest(score_func=chi2, k='all') # k='all' 表示保留所有特征分数，不进行删减
selector.fit(X_scaled, y)

scores = selector.scores_
pvalues = selector.pvalues_

# ==========================================
# 5. 生成结果表格 (Weka 风格)
# ==========================================

results = []
for i, col in enumerate(X.columns):
    score = scores[i]
    pval = pvalues[i]
    category = feature_source_map.get(col, 'Unknown') # 获取该特征原本属于哪个文件
    
    results.append({
        'Feature': col,
        'Chi2_Score': score,
        'P_Value': pval,
        'Category': category
    })

df_results = pd.DataFrame(results)

# 按 Chi2 分数降序排列 (Contribution High -> Low)
df_results = df_results.sort_values(by='Chi2_Score', ascending=False)

# 添加排名列 (Rank)
df_results.reset_index(drop=True, inplace=True)
df_results.index += 1 
df_results.index.name = 'Rank'
df_results.reset_index(inplace=True)

# ==========================================
# 6. 保存与展示
# ==========================================

output_file = 'Chi2_Feature_Contribution.csv'
df_results.to_csv(output_file, index=False)

print("\n=== Top 20 Features by Chi2 Contribution ===")
print(df_results.head(20).to_string(index=False))

print(f"\n完整结果已保存至: {output_file}")

# 额外分析：查看哪个类别的特征总体贡献最大（平均分）
print("\n=== Average Contribution by Feature Category ===")
category_stats = df_results.groupby('Category')['Chi2_Score'].mean().sort_values(ascending=False)
print(category_stats)