import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 路径配置 (Path Configuration)
# 请将 DATA_DIR 修改为你的数据目录路径
# 原始路径: D:\OneDrive\...\peter pan txt\二轮大修\data
# ==========================================
DATA_DIR = ""  # 示例：r"D:\path\to\data"
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from functools import reduce

# ==========================================
# 1. 配置区域 (Configuration) - 对应要求 6
# ==========================================

# 模式选择: 'full' (所有特征) 或 'selected' (自定义特征列表)
FEATURE_MODE = 'full'  # 可选 'full' 或 'selected'

# 如果选择了 'selected'，请在这里填入你筛选后的特征列名 list
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
    # 在这里添加更多特征...
]

# ==========================================
# 2. 数据读取与预处理 (Data Loading & Preprocessing)
# ==========================================

def load_and_merge_data():
    df_dep   = pd.read_csv(os.path.join(DATA_DIR, 'DepTag_result.csv'))
    df_lex   = pd.read_csv(os.path.join(DATA_DIR, 'Lexical&Translatibility_features.csv'))
    df_ngram = pd.read_csv(os.path.join(DATA_DIR, 'N_gram_features.csv'))
    df_read  = pd.read_csv(os.path.join(DATA_DIR, 'readibility_results.csv'))

    # --- 数据清洗 ---
    # 观察发现 readibility_results.csv 的文件名带有 "_pos.txt" 后缀，需要去掉才能匹配
    # df_read['file_name'] = df_read['file_name'].str.replace('_pos.txt', '', regex=False)
    
    # 统一去除可能存在的空格
    dfs = [
        df_dep, 
           df_lex, 
           df_ngram, 
           df_read
        ]
    for df in dfs:
        if 'file_name' in df.columns:
            df['file_name'] = df['file_name'].str.strip()

    # --- 合并数据 (Merge) ---
    # 对应要求 1: All features implies merging all
    df_final = reduce(lambda left, right: pd.merge(left, right, on='file_name', how='inner'), dfs)
    
    return df_final

# 加载数据
data = load_and_merge_data()

# --- 提取标签 (Label Extraction) ---
# 假设文件名格式为 "Author_Chapter"，如 "Liang1929_10"。标签应该是 "Liang1929"
# 我们取第一个下划线前的部分作为类别标签
data['label'] = data['file_name'].apply(lambda x: x.split('_')[0])

print(f"Total samples: {len(data)}")
print(f"Total features (raw): {data.shape[1] - 2}") # -2 因为排除了 file_name 和 label
print(f"Classes found: {data['label'].unique()}")

# ==========================================
# 3. 特征筛选 (Feature Selection Logic)
# ==========================================

# 排除非特征列
non_feature_cols = ['file_name', 'label']
all_features = [c for c in data.columns if c not in non_feature_cols]

if FEATURE_MODE == 'selected':
    # 检查选定特征是否都在数据中
    valid_features = [f for f in SELECTED_FEATURES_LIST if f in all_features]
    print(f"\n[Mode: Selected] Using {len(valid_features)} features.")
    if len(valid_features) < len(SELECTED_FEATURES_LIST):
        print("Warning: Some selected features were not found in the CSVs.")
    X = data[valid_features]
else:
    print(f"\n[Mode: Full] Using all {len(all_features)} features.")
    X = data[all_features]

y = data['label']

# --- 数据标准化 (Standardization) ---
# SVM 和 Logistic Regression 对数据尺度敏感，必须标准化
SCALE_DATA = True  # 添加开关以控制是否进行标准化

if SCALE_DATA:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(0)) # 填充NaN为0以防万一
else:
    X_scaled = X.fillna(0).values  # 如果不开启标准化，直接使用原始数据

# ==========================================
# 4. 模型定义 (Model Definitions) - 对应要求 2
# ==========================================

# Linear SVM (probability=True is needed for ensemble averaging)
svm_clf = SVC(kernel='linear', probability=True, random_state=42)

# Simple Logistic Regression
lr_clf = LogisticRegression(max_iter=1000, random_state=42)

# Naive Bayes
nb_clf = GaussianNB()

# ==========================================
# 5. 交叉验证与集成 (CV & Ensemble) - 对应要求 3, 4
# ==========================================

# 10-fold Cross-Validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("\nRunning 10-fold Cross-Validation and Generating Probabilities...")

# 使用 cross_val_predict 获取所有样本在作为测试集时的预测概率
# method='predict_proba' 返回每个类别的概率
try:
    # 1. SVM Probabilities
    probas_svm = cross_val_predict(svm_clf, X_scaled, y, cv=cv, method='predict_proba', n_jobs=-1)
    
    # 2. Logistic Regression Probabilities
    probas_lr = cross_val_predict(lr_clf, X_scaled, y, cv=cv, method='predict_proba', n_jobs=-1)
    
    # 3. Naive Bayes Probabilities
    probas_nb = cross_val_predict(nb_clf, X_scaled, y, cv=cv, method='predict_proba', n_jobs=-1)
    
except Exception as e:
    print(f"Error during cross-validation: {e}")
    exit()

# --- 集成 (Ensemble): Averaging Probabilities ---
# 对应要求 4: averaging classifiers’ performance
avg_probas = (probas_svm + probas_lr + probas_nb) / 3

# 获取最终预测类别 (概率最大的那个)
class_labels = svm_clf.fit(X_scaled, y).classes_ # 获取类别名称列表
y_pred_indices = np.argmax(avg_probas, axis=1)
y_pred = class_labels[y_pred_indices]

# ==========================================
# 6. 生成结果表格 (Result Table) - 对应要求 4, 图表展示
# ==========================================

# 构建结果 DataFrame
results_df = pd.DataFrame()
results_df['File Name'] = data['file_name']
results_df['True Label'] = data['label']

# 将概率添加到表中
for i, label in enumerate(class_labels):
    results_df[f'Prob_{label}'] = avg_probas[:, i]

results_df['Predicted Label'] = y_pred
results_df['Correct?'] = results_df['True Label'] == results_df['Predicted Label']

# 打印类似图中表格的前几行
print("\n=== Classification Results (Ensemble) ===")
# 格式化输出，保留4位小数
print(results_df.round(4).head(10).to_string()) 

# 保存完整表格到CSV
results_df.to_csv('Ensemble_Classification_Results.csv', index=False)
print("\n(Full result table saved to 'Ensemble_Classification_Results.csv')")

# ==========================================
# 7. 混淆矩阵 (Confusion Matrix) - 对应要求 5
# ==========================================

cm = confusion_matrix(y, y_pred, labels=class_labels)
acc = accuracy_score(y, y_pred)

print(f"\nFinal Ensemble Accuracy: {acc:.4f}")

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.title(f'Confusion Matrix (Ensemble 10-Fold)\nAccuracy: {acc:.2%}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()



# 打印详细分类报告
print("\n=== Detailed Classification Report ===")
report = classification_report(y, y_pred)
print(report)

# 将分类报告保存到CSV文件
classification_report_dict = classification_report(y, y_pred, output_dict=True,digits=2)

# 计算每个类的准确率（acc）并添加到报告中
for label in class_labels:
    true_positive = cm[class_labels.tolist().index(label), class_labels.tolist().index(label)]
    total_samples = cm[class_labels.tolist().index(label), :].sum()
    class_acc = round(true_positive / total_samples, 4) if total_samples > 0 else 0
    classification_report_dict[label]['accuracy'] = class_acc

classification_report_df = pd.DataFrame(classification_report_dict).transpose()
classification_report_df = classification_report_df.round(2)  # 保留2位小数
classification_report_df.to_csv('Classification_Report.csv', index=True)
print("\n(Classification report saved to 'Classification_Report.csv')")