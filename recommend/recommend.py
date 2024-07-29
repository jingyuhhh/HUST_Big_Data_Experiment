import random
import pandas as pd
from math import log
import numpy as np
from scipy.spatial.distance import cosine
import argparse
import sys
import math
import time

def calculate_pearson_similarity(user_i, user_j):
    common_anime = set(user_i.keys()) & set(user_j.keys())
    if len(common_anime) == 0:
        return 0
    else:
        user_i_mean = np.mean(list(user_i.values()))
        user_j_mean = np.mean(list(user_j.values()))
        numerator = sum((user_i.get(a, 0) - user_i_mean) * (user_j.get(a, 0) - user_j_mean) for a in common_anime)
        denominator = np.sqrt(sum((user_i.get(a, 0) - user_i_mean) ** 2 for a in user_i)) * \
                      np.sqrt(sum((user_j.get(a, 0) - user_j_mean) ** 2 for a in user_j))
        return 0 if denominator == 0 else numerator / denominator

def get_user_similar(user_anime_matrix, user_id_all):
    try:
        return np.load('./sim/user_similar.npy')
    except:
        user_num = max(user_id_all)
        user_similarity_matrix = np.zeros((user_num + 1, user_num + 1))
        for i in user_anime_matrix:
            for j in user_anime_matrix:
                if i <= j:
                    user_similarity_matrix[i][j] = user_similarity_matrix[j][i] = calculate_pearson_similarity(
                        user_anime_matrix[i], user_anime_matrix[j])
        np.save('./sim/user_similar.npy', user_similarity_matrix)
        return user_similarity_matrix

def generate_affine_hash_function(n):
    while True:
        a = random.randint(1, n-1)
        if math.gcd(a, n) == 1:
            b = random.randint(0, n-1)
            return lambda x: (a * x + b) % n

def generate_affine_hash_functions(n, num_functions):
    hash_functions = []
    for _ in range(num_functions):
        hash_functions.append(generate_affine_hash_function(n))
    return hash_functions

def generate_hash_matrix(anime_num, num_hash):
    hash_matrix = np.zeros((anime_num, num_hash), dtype=int)
    hash_functions = generate_affine_hash_functions(anime_num, num_hash)
    for i in range(anime_num):
        for j in range(num_hash):
            hash_matrix[i][j] = hash_functions[j](i)
    return hash_matrix

def calculate_jaccard_similarity(user_minhash_matrix, user_num, num_hash):
    user_similarity_matrix = np.eye(user_num+1) # 对角线为1
    for i in range(1, user_num+1):
        for j in range(i+1, user_num):
            # 计算相似度，即两个用户的最小哈希值相等的数量
            num = sum(user_minhash_matrix[k][i] == user_minhash_matrix[k][j] for k in range(num_hash))
            user_similarity_matrix[i][j] = user_similarity_matrix[j][i] = num/num_hash
        if i % 1000 == 0:
            print(f"已完成{i / user_num * 100:.2f}%")
    return user_similarity_matrix

def get_user_similar2(user_anime_matrix, anime_id_all, user_id_all):
    try:
        return np.load('./sim/user_similar_2.npy')
    except:
        user_num = max(user_id_all)
        anime_num = len(anime_id_all)
        anime_index = {anime_id: index for index, anime_id in enumerate(anime_id_all)}
        # 01 处理效用矩阵
        user_anime_matrix_01 = {user_id: {anime_id: 1 if score >= 5 else 0 for anime_id, score in anime_dict.items()} for user_id, anime_dict in user_anime_matrix.items()}
        # 生成 hash 矩阵
        hash_matrix = generate_hash_matrix(anime_num, 100)
        user_minhash_matrix = np.zeros((100, user_num+1))
        for user_id, anime_dict in user_anime_matrix_01.items():
            temp_min_array = [random.randint(100001, sys.maxsize)]*100
            for anime_id, score in anime_dict.items():
                if score == 1:
                    # 更新最小哈希值
                    temp_min_array = [min(hash_matrix[anime_index[anime_id]][i], temp_min_array[i]) for i in range(100)]
            user_minhash_matrix[:, user_id] = temp_min_array
        user_similarity_matrix = calculate_jaccard_similarity(user_minhash_matrix, user_num, 100)
        np.save('./sim/user_similar_2.npy', user_similarity_matrix)
        return user_similarity_matrix

def tf_idf(genres, all_genres, all_genres_count, num_anime, num_genres, genre_map):
    feature_matrix = np.zeros((num_anime, num_genres))
    for i in range(num_anime):
        anime_genres = genres[i]
        for index in range(num_genres):
            genre = all_genres[index]
            # 计算 TF
            tf = anime_genres.count(genre) / len(anime_genres)
            # 计算 IDF
            doc_count = all_genres_count[index]
            idf = log(num_anime / doc_count, 2)
            # 计算 TF-IDF
            feature_matrix[i, genre_map[genre]] = tf * idf
    return feature_matrix

def get_content_similar():
    df = pd.read_csv('./data/anime.csv')
    anime_ids = df['Anime_id'].tolist()
    try:
        content_similarity_matrix = np.load('./sim/content_similar.npy'), anime_ids
        return content_similarity_matrix
    except:
        # 读取数据集
        df = pd.read_csv('./data/anime.csv')
        # 提取动漫类别作为特征值
        genres = df['Genres'].str.split(', ').tolist()
        all_genres = list(set([genre for genre_list in genres for genre in genre_list]))
        genre_map = {genre: i for i, genre in enumerate(all_genres)}
        all_genres_count = [sum(genre == g for genre_list in genres for g in genre_list) for genre in all_genres]

        num_genres = len(all_genres)
        num_anime = len(df)

        # 计算 TF-IDF 特征矩阵
        feature_matrix = tf_idf(genres, all_genres, all_genres_count, num_anime, num_genres, genre_map)
        # 计算动漫之间的相似度矩阵
        content_similarity_matrix = np.eye(num_anime, dt_ype=float)
        for i in range(num_anime):
            for j in range(i + 1, num_anime):
                content_similarity_matrix[i, j] = content_similarity_matrix[j, i] = 1 - cosine(feature_matrix[i],
                                                                                               feature_matrix[j])
            if i % 1000 == 0:
                print(f"已完成{i / num_anime * 100:.2f}%")
        # 存储相似度矩阵
        np.save('./sim/content_similar.npy', content_similarity_matrix)
        return content_similarity_matrix, anime_ids


def get_content_similar2():
    df = pd.read_csv('./data/anime.csv')
    anime_ids = df['Anime_id'].tolist()
    try:
        content_similarity_matrix = np.load('./sim/content_similar_2.npy'), anime_ids
        return content_similarity_matrix
    except:
        # 读取数据集
        df = pd.read_csv('./data/anime.csv')
        # 提取动漫类别作为特征值
        genres = df['Genres'].str.split(', ').tolist()
        all_genres = list(set([genre for genre_list in genres for genre in genre_list]))
        num_genres = len(all_genres)
        num_anime = len(df)
        # 01 处理特征矩阵， 行代表动漫，列代表类别， 1 代表包含， 0 代表不包含
        feature_matrix_01 = np.array([[1 if genre in genre_list else 0 for genre in all_genres] for genre_list in genres])

        num_hash = 20
        hash_functions = generate_affine_hash_functions(num_genres,num_hash)
        hash_matrix = np.array([hash_functions[j](i) for j in range(num_hash) for i in range(num_genres)]).reshape(
            num_genres, num_hash)

        # 计算最小哈希矩阵
        content_minhash_matrix = np.zeros((num_hash,num_anime))
        for i in range(num_anime):
            temp_min_array = [random.randint(100001, sys.maxsize)]*num_hash
            for j in range(num_genres):
                if feature_matrix_01[i][j] == 1:
                    # 更新最小哈希值
                    for k in range(num_hash):
                        temp_min_array[k] = min(hash_matrix[j][k], temp_min_array[k])
            for j in range(num_hash):
                content_minhash_matrix[j][i] = temp_min_array[j]

        # 计算动漫之间的相似度矩阵
        content_similarity_matrix = calculate_jaccard_similarity(content_minhash_matrix, num_anime, num_hash)
        # 存储相似度矩阵
        np.save('./sim/content_similar_2.npy', content_similarity_matrix)
        return content_similarity_matrix, anime_ids

def get_user_predict_test(user_similarity_matrix: np.ndarray, k=150):
    # 读取测试集数据
    user_id, anime_id, rating, _ = get_data('./data/test_set.csv')

    predictions = {}
    SSE = 0
    for user_id, anime_id, rating in zip(user_id, anime_id, rating):
        # 找到与当前用户最相似的 k 个用户
        similar_users = list(np.sort(user_similarity_matrix[user_id], kind='quicksort')[::-1][:k])
        sorted_indices = list(np.argsort(user_similarity_matrix[user_id])[::-1][:k])
        # 计算预测评分
        weighted_sum = 0
        similarity_sum = 0
        for similar_user, similarity in zip(sorted_indices, similar_users):
            if anime_id in user_anime_matrix[similar_user]:
                weighted_sum += similarity * user_anime_matrix[similar_user][anime_id]
                similarity_sum += similarity
        if similarity_sum > 0:
            predictions[(user_id, anime_id)] = round(weighted_sum / similarity_sum)
        else:
            predictions[(user_id, anime_id)] = 0
        SSE += (predictions[(user_id, anime_id)] - rating) ** 2

    return predictions, SSE


def get_user_predict(user_id, user_similarity_matrix: np.ndarray, anime_id_all, n=20, k=500):
    # 找到与当前用户最相似的 k 个用户
    similar_users = list(np.sort(user_similarity_matrix[user_id], kind='quicksort')[::-1][:k])
    sorted_indices = list(np.argsort(user_similarity_matrix[user_id])[::-1][:k])
    predictions = {}
    for anime_id in anime_id_all:
        # 计算预测评分
        weighted_sum = 0
        similarity_sum = 0
        for similar_user, similarity in zip(sorted_indices, similar_users):
            if anime_id in user_anime_matrix[similar_user]:
                weighted_sum += similarity * user_anime_matrix[similar_user][anime_id]
                similarity_sum += similarity
        if similarity_sum > 0:
            predictions[anime_id] = weighted_sum / similarity_sum
        else:
            predictions[anime_id] = 0
    seen = set()
    predictions_items = [item for item in sorted(predictions.items(), key=lambda x: x[1], reverse=True) if item[1] not in seen and not seen.add(item[1])]
    return dict(predictions_items[:n])



def get_content_predict_test(content_similarity_matrix: np.ndarray, user_anime_matrix: np.ndarray, anime_ids):
    # 读取测试集数据
    user_id, anime_id, rating, _ = get_data('./data/test_set.csv')

    predictions = {}
    SSE = 0
    for user_id, anime_id, rating in zip(user_id, anime_id, rating):
        # 获取当前用户已打分的动漫的集合
        anime_has_rated = user_anime_matrix[user_id]
        anime_sims = list(content_similarity_matrix[anime_ids.index(anime_id)])
        weighted_sum = 0
        similarity_sum = 0
        for key, value in anime_has_rated.items():
            anime_sim = anime_sims[anime_ids.index(key)]
            if anime_sim > 0:
                similarity_sum += anime_sim
                weighted_sum += anime_sim * value
        if similarity_sum > 0:
            predictions[(user_id, anime_id)] = round(weighted_sum / similarity_sum)
        else:
            predictions[(user_id, anime_id)] = 0
        SSE += (predictions[(user_id, anime_id)] - rating) ** 2
    return predictions, SSE


def get_content_predict(user_id, content_similarity_matrix: np.ndarray, user_anime_matrix: np.ndarray, anime_ids,
                        n=20):
    predictions = {}
    # 获取当前用户已打分的动漫的集合
    anime_has_rated = user_anime_matrix[user_id].items()

    # 获取这些已打分动漫的相似度集合
    anime_sims = {anime_id: list(content_similarity_matrix[anime_ids.index(anime_id)]) for anime_id, rating in
                  anime_has_rated}

    total_num = len(anime_ids)
    for anime_id, index in zip(anime_ids, range(total_num)):
        if anime_id in anime_has_rated:
            continue
        weighted_sum = 0
        similarity_sum = 0
        for key, value in anime_has_rated:
            anime_sim = anime_sims[key][index]
            if anime_sim > 0:
                similarity_sum += anime_sim
                weighted_sum += anime_sim * value
        if similarity_sum > 0:
            predictions[anime_id] = weighted_sum / similarity_sum
        else:
            predictions[anime_id] = 0
        if index % 1000 == 0:
            print(f"已完成{index / total_num * 100:.2f}%")
    seen = set()
    predictions_items = [item for item in sorted(predictions.items(), key=lambda x: x[1], reverse=True) if
                         item[1] not in seen and not seen.add(item[1])]
    return dict(predictions_items[:n])


def get_data(file_name):
    df = pd.read_csv(file_name)
    df['user_id'] = df['user_id'].astype(int)
    df['anime_id'] = df['anime_id'].astype(int)
    df['rating'] = df['rating'].astype(int)
    user_id_all = set(df['user_id'])
    anime_id_all = set(df['anime_id'])
    ratings = set(df['rating'])

    return user_id_all, anime_id_all, ratings, df


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='collaborative', help='collaborative or content')
    parser.add_argument('--level', type=str, default='simple', help='simple or enhanced')
    parser.add_argument('--user_id', type=int, default=1, help='user id')
    args = parser.parse_args()

    # 读取训练集数据
    user_id_all, anime_id_all, _, train_data = get_data('./data/train_set.csv')
    user_anime_matrix = train_data.groupby('user_id').apply(lambda x: dict(zip(x['anime_id'], x['rating']))).to_dict()

    if args.mode == 'collaborative':
        if args.level == 'simple':
            user_similarity_matrix = get_user_similar(user_anime_matrix, user_id_all)
        else:
            user_similarity_matrix = get_user_similar2(user_anime_matrix, anime_id_all, user_id_all)
        pre, SSE = get_user_predict_test(user_similarity_matrix)
        predictions = get_user_predict(args.user_id, user_similarity_matrix, anime_id_all)
    else:
        if args.level == 'simple':
            content_similarity_matrix, anime_ids = get_content_similar()
        else:
            content_similarity_matrix, anime_ids = get_content_similar2()
        pre, SSE = get_content_predict_test(content_similarity_matrix, user_anime_matrix, anime_ids)
        predictions = get_content_predict(args.user_id, content_similarity_matrix, user_anime_matrix, anime_ids)

    end = time.time()

    with open(f'./result/{args.mode}_{args.level}.txt', 'w') as f:
        f.write(f"用户{args.user_id}推荐结果：\n")
        for k, v in predictions.items():
            f.write(f"{k}: {v:.2f}\n")
        f.write(f"耗时：{end-start}秒\n")
        f.write(f"SSE: {SSE}\n")
