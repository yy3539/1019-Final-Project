import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np  # 👈 新加的库

# 读取数据（原始6列: label, id, date, query, user, text）
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
df.columns = ['label', 'id', 'date', 'query', 'user', 'text']

# 只保留正负类（0=negative, 4=positive）
df = df[df['label'] != 2]
df['label'] = df['label'].replace({0: 0, 4: 1})

# 特征提取: TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 打印 top 20 TF-IDF 关键词 🔍
feature_names = vectorizer.get_feature_names_out()
tfidf_means = X.mean(axis=0).A1
top_indices = np.argsort(tfidf_means)[::-1][:20]

print("Top 20 keywords by average TF-IDF score:")
for i in top_indices:
    print(f"{feature_names[i]}: {tfidf_means[i]:.4f}")

# 训练模型 + 5折交叉验证
model = LogisticRegression(max_iter=1000)
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"Cross-validation Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
