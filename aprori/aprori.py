import ast
import json
import os
import time
from itertools import combinations

min_support = 0.15  # 频繁项集最小支持度
min_confidence = 0.3  # 关联规则最小置信度


def timer(func):
    """计时装饰器函数"""
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('%s cost time: %.3f s' % (func.__name__, time_spend))
        return result

    return func_wrapper

# 生成候选频繁一项集
@timer
def generate_c1(data_set: list[list]) -> set:
    """生成候选频繁一项集

    Args:
        data_set (list[list]): _description_

    Returns:
        set: 频繁一项集
    """
    c1 = set()
    for basket in data_set:
        for item in basket:
            # frozenset:冻结集合，可以作为字典的key
            item = frozenset([item])
            c1.add(item)
    return c1

@timer
def generate_lk(data_set: list[list], ck: set[frozenset], min_support: float = min_support) \
        -> tuple[set[frozenset], dict[frozenset, float]]:
    """根据 Ck 生成频繁 k 项集 Lk

    Args:
        data_set (list[list]): 数据集
        ck (set[frozenset]): k 阶候选频繁项集
        min_support (float, optional): 最小支持度. 默认 0.005.

    Returns:
        tuple[set[frozenset], dict[frozenset, float]]: 频繁 k 项集, 频繁项集支持度
    """
    # 字典，记录每个候选频繁项集出现的次数
    item_set_count = {}
    # 频繁k项集
    lk = set()
    # 记录频繁项集的支持度
    fre_item_set_sup = {}
    for basket in data_set:
        for item_set in ck:
            if item_set.issubset(basket):
                # 统计候选频繁项集出现次数
                if item_set in item_set_count:
                    item_set_count[item_set] += 1
                else:
                    item_set_count[item_set] = 1
    data_num = len(data_set)
    print("yes")
    for key, value in item_set_count.items():
        # 计算支持度
        support = value / data_num
        if support >= min_support:
            print(key, support)
            lk.add(key)
            fre_item_set_sup[key] = support

    return lk, fre_item_set_sup

def is_k_sub(k_item_set: frozenset, lk: set) -> bool:
    """判断 k 项集是否是 k + 1 项子集

    Args:
        k_item_set (frozenset): k 项集
        lk (set): k 阶频繁项集

    Returns:
        bool: k 项集是否是 k + 1 项子集
    """
    for item in k_item_set:
        sub_item = k_item_set - frozenset([item])
        if sub_item not in lk:
            return False
    return True

# @timer
# def generate_next_ck(lk: set[frozenset], k: int) -> set[frozenset]:
#     """根据 Lk 构造候选频繁项集 Ck+1
#
#     Args:
#         lk (set[frozenset]): k 阶频繁项集
#         k (int): 频繁项集基数
#
#     Returns:
#         set[frozenset]: 频繁候选项集 Ck+1
#     """
#     ck = set()
#     hash_table = {}
#     for set1 in lk:
#         for set2 in lk:
#             union_set = set1 | set2
#             if len(union_set) == k:
#                 is_k_sub = True
#                 # 判断是否是 k + 1 项子集
#                 for item in union_set:
#                     sub_item = union_set - frozenset([item])
#                     if sub_item not in lk:
#                         is_k_sub = False
#                         break
#                 if is_k_sub:
#                     ck.add(union_set)
#                     # if union_set in hash_table:
#                     #     hash_table[union_set] += 1
#                     # else:
#                     #     hash_table[union_set] = 1
#     # 使用 PCY 优化
#     # for key, value in hash_table.items():
#     #     print(key, value)
#     #     if value >= min_support * len(dataset):
#     #         ck.add(key)
#
#     return ck

@timer
def generate_next_ck(lk: set[frozenset], k: int) -> set[frozenset]:
    """根据 Lk 构造候选频繁项集 Ck+1

    Args:
        lk (set[frozenset]): k 阶频繁项集
        k (int): 频繁项集基数

    Returns:
        set[frozenset]: 频繁候选项集 Ck+1
    """
    ck = set()
    for set1 in lk:
        for set2 in lk:
            union_set = set1 | set2
            # * 剪枝策略和连接策略
            if len(union_set) == k and is_k_sub(union_set, lk):
                ck.add(union_set)
    return ck


@timer
def generate_rules(l3: set[frozenset], sup1: dict[frozenset, float], sup2: dict[frozenset, float],
                   sup3: dict[frozenset, float], sup4: dict[frozenset, float], min_confidence: float = min_confidence):
    """生成符合置信度要求的关联规则

    Args:
        l3 (set[frozenset]): 四阶频繁项集
        sup1 (dict[frozenset, float]): 一阶频繁项集支持度
        sup2 (dict[frozenset, float]): 二阶频繁项集支持度
        sup3 (dict[frozenset, float]): 三阶频繁项集支持度
        sup4 (dict[frozenset, float]): 四阶频繁项集支持度
        min_confidence (float, optional): 最小置信度要求

    Returns:
        list: 关联规则
    """
    rule_list = []
    for fre_item_set in l3:
        union_sup = sup3[fre_item_set]
        # 1 =>3, 3=> 1
        for k in range(4):
            tmp = list(fre_item_set)
            set1 = [tmp[k]]
            tmp.remove(tmp[k])
            set2 = tmp
            conf12 = union_sup / sup1[frozenset(set1)]
            conf21 = union_sup / sup2[frozenset(set2)]
            if conf12 >= min_confidence:
                rule_list.append((set(set1), set(set2), conf12))
            if conf21 >= min_confidence:
                rule_list.append((set(set2), set(set1), conf21))
        # 2 => 2
        for set1 in combinations(fre_item_set, 2):
            set2 = set(fre_item_set) - set(set1)
            conf12 = union_sup / sup2[frozenset(set(set1))]
            if conf12 >= min_confidence:
                rule_list.append((set(set1), set(set2), conf12))
    return rule_list


def save_fre_item_set(filename: str, fre_item_set_sup: dict[frozenset, float]) -> None:
    """保存频繁项集

    Args:
        filename (str): 文件名
        fre_item_set_sup (dict[frozenset, float]): 支持度
    """
    f_write = open(os.path.join("./", filename), 'w')
    f_write.write('{},\t{}\n'.format('frequent-itemSets', 'support'))
    for k, v in fre_item_set_sup.items():
        f_write.write("{},\t{}\n".format(set(k), v))
    print("{} done.".format(filename))


def save_rules(filename: str, rule_list: list) -> None:
    """保存关联规则

    Args:
        filename (str): 文件名
        rule_list (list): 关联规则
    """
    f_write = open(os.path.join("./", filename), 'w')
    for rule in rule_list:
        f_write.write("{} => {}, {}\n".format(rule[0], rule[1], rule[2]))


if __name__ == '__main__':
    start = time.time()
    dataset = []
    with open('../mapreduce/reduce_result/result.txt', 'r') as f:
        # [word count [references]]
        # --> [word references]
        # 前三行
        for i, line in enumerate(f):
            # if i > 2:
            #     break
            tmp = []
            parts = line.split()
            tmp.append(parts[0])
            tmp.extend(ast.literal_eval(' '.join(parts[2:])))
            dataset.append(tmp)
    # 生成频繁一项集
    C1 = set()
    for basket in dataset:
        for item in basket:
            # print(item)
            # frozenset:冻结集合，可以作为字典的key
            item = frozenset([item])
            C1.add(item)
    # 生成频繁项集
    L1, fre_item_set_sup1 = generate_lk(dataset, C1)
    save_fre_item_set("L1.csv", fre_item_set_sup1)
    C2 = generate_next_ck(L1, 2)
    L2, fre_item_set_sup2 = generate_lk(dataset, C2)
    save_fre_item_set("L2.csv", fre_item_set_sup2)
    C3 = generate_next_ck(L2, 3)
    L3, fre_item_set_sup3 = generate_lk(dataset, C3)
    save_fre_item_set("L3.csv", fre_item_set_sup3)
    C4 = generate_next_ck(L3, 4)
    L4, fre_item_set_sup4 = generate_lk(dataset, C4)
    save_fre_item_set("L4.csv", fre_item_set_sup4)
    rules_list = generate_rules(
        L4, fre_item_set_sup1, fre_item_set_sup2, fre_item_set_sup3, fre_item_set_sup4)
    end = time.time()


    save_rules("rule", rules_list)

    print("1阶频繁项集个数为:", len(L1))
    print("2阶频繁项集个数为:", len(L2))
    print("3阶频繁项集个数为:", len(L3))
    print("关联规则个数为:", len(rules_list))
    print("用时:", end - start, 's')