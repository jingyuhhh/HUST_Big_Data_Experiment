import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler



def process_data(filename: str, k: int, samples_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    """处理数据

    Args:
        filename (str): 数据文件名
        k (int): 聚类中心数量
        samples_per_class (int): 每个类别的样本数量

    Returns:
        tuple[np.ndarray, np.ndarray]: 数据和标签
    """
    df = pd.read_csv(filename)
    df.drop(df.columns[-3:], axis=1, inplace=True)
    # 删掉 Unkown 的行
    df = df.replace('Unknown', np.nan)
    df = df.dropna(axis=0)
    df.drop(['Anime_id', 'Name', 'Genres', 'Ranked'], axis=1, inplace=True)
    df = df.sort_values(by='Popularity', ascending=False)
    df['Popularity'] = pd.to_numeric(df['Popularity'], errors='coerce')

    df['Popularity_Label'] = pd.qcut(df['Popularity'], k, labels=False)
    df_final = pd.DataFrame()
    for i in range(k):
        df_i = df[df['Popularity_Label'] == i].head(samples_per_class)
        df_final = pd.concat([df_final, df_i])

    columns_to_normalize = ['Score-2', 'Score-3', 'Score-4', 'Score-5', 'Score-6', 'Score-7', 'Score-8', 'Score-9',
                            'Score-10', 'Popularity']
    scaler = StandardScaler()
    df_final[columns_to_normalize] = scaler.fit_transform(df_final[columns_to_normalize])
    print(df_final.head())
    return df_final.drop(['Popularity_Label'], axis=1).values, df_final['Popularity_Label'].values

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2, axis=1))

def kmeans(data, k):
    # 1. 随机选择初始质心
    centers = data[np.random.choice(range(data.shape[0]), size=k, replace=False)]

    while True:
        # 2. 分配标签基于最近的中心
        distances = np.array([euclidean_distance(center, data) for center in centers])
        labels = np.argmin(distances, axis=0)

        # 3. 找到新的中心
        new_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # 4. 检查收敛
        if np.all(centers == new_centers):
            break
        centers = new_centers
    # 计算所有数据点到各自质心距离的平方和
    distances = np.array([euclidean_distance(center, data) for center in centers])
    distances = distances.sum()
    print("Distances: ", distances)
    return centers, labels


if __name__ == "__main__":
    # 处理数据
    k = 3
    samples_per_class = 60
    data, labels = process_data("anime.csv", k, samples_per_class)
    centers, predict_labels = kmeans(data,k)
    accuracy = np.mean(labels == predict_labels)
    print("Accuracy: ", accuracy)
    score10_index = 1
    score2_index = 9
    score10_data = data[:, score10_index]
    score2_data = data[:, score2_index]

    # 绘制聚类结果
    plt.figure(figsize=(10, 8))
    for i in range(k):
        plt.scatter(score10_data[labels == i], score2_data[labels == i], label=f'Cluster {i + 1}')
    plt.scatter(centers[:, score10_index], centers[:, score2_index], s=300, c='red', label='Centroids')
    plt.xlabel('Score-10')
    plt.ylabel('Score-2')
    plt.title('K-means Clustering')
    plt.legend()
    plt.show()
