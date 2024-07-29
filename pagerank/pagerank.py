import numpy as np
import ast


def create_adjacency_matrix(data):
    # 获取节点数量
    n = len(data)
    # 创建一个 n*n 的全零矩阵
    adjacency_matrix = np.zeros((n, n))
    # 获取所有标题
    titles = [i[0] for i in data]
    # 创建一个标题到索引的映射
    title_to_index = {title: index for index, title in enumerate(titles)}

    # 遍历数据，添加边
    for title, references in data:
        for reference in references:
            reference = reference.split(".")[0]
            if reference in title_to_index:
                adjacency_matrix[title_to_index[title]][title_to_index[reference]] = 1
    column_sum = adjacency_matrix.sum(axis=0)
    # 将列的和归一化为 1
    for i in range(n):
        if column_sum[i] != 0:
            adjacency_matrix[:, i] /= column_sum[i]
    return adjacency_matrix, titles


def pagerank(data, beta=0.85, tol=1e-8):
    # 创建邻接矩阵
    adjacency_matrix, titles = create_adjacency_matrix(data)
    print(adjacency_matrix)
    n = len(adjacency_matrix)
    # 初始化 pagerank 值
    pagerank_value = np.ones(n) / n
    A = (1 - beta) * np.ones((n, n)) / n + beta * adjacency_matrix
    while True:
        new_pagerank_value = np.dot(A, pagerank_value)
        new_pagerank_value /= new_pagerank_value.sum()
        # 如果新旧 pagerank 值之间的差异小于预设的阈值，则停止迭代
        if np.abs(new_pagerank_value - pagerank_value).sum() < tol:
            return {titles[i]: rank for i, rank in enumerate(new_pagerank_value)}
        pagerank_value = new_pagerank_value


with open("../mapreduce/reduce_result/top_1000.txt", 'r') as file:
    # word count [references]
    # 使用 ast.literal_eval 将字符串转换为元组
    data = []
    for line in file:
        parts = line.split()
        data.append((parts[0], ast.literal_eval(' '.join(parts[2:]))))

rankings = pagerank(data)
with open("rankings.txt", 'w') as file:
    for title, rank in rankings.items():
        file.write(f'{title} {rank}\n')


