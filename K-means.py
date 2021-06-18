# -*- coding: utf-8 -*
# @Time: 2021/6/3 13:40
# @File : K-means.py
# 西瓜书习题9.4 编程实现K均值算法

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def k_means(X, n_class=3):
    X_len = len(X)
    index = list(range(len(X)))
    random.shuffle(index)
    random_init_avg_vec_index = index[:n_class]   # 从X中随机选择n_class个样本作为初始均值向量
    print(str(n_class)+"个中心位置分别为:", random_init_avg_vec_index)
    random_init_avg_vec = X[random_init_avg_vec_index, :]
    clusters = {}   # 以字典的形式保存每个簇

    for i in range(n_class):
        clusters[i] = [random_init_avg_vec[i].tolist()]     # 初始化每个簇的中心向量
    old_clusters = {0: np.random.normal(0, 0.01, (1, 2))}   # 随机生成一组参考中心

    while old_clusters != clusters:         # 只要每个簇的中心还在变化，就继续迭代

        old_clusters = clusters.copy()
        clusters = {}
        for i in range(n_class):
            clusters[i] = [random_init_avg_vec[i].tolist()]  # 记录每个簇的中心向量

        for i in range(X_len):      # 计算每个样本与均值向量之间的距离
            dists = []
            for item in random_init_avg_vec:
                dist = np.sqrt(((X[i] - item) ** 2).sum())  # 计算样本和均值向量的闵可夫斯基距离
                dists.append(dist)
            clusters[int(np.argmin(dists))].append(X[i].tolist())   # 根据最近均值向量确定样本的簇标记
        for key in clusters.keys():
            array = np.asarray(clusters[key])
            random_init_avg_vec[key] = array.sum(axis=0) / len(array)   # 更新每个簇的均值向量
    return clusters


def plot_k_means(clusters):
    mid = []
    for key in clusters.keys():
        mid.append(clusters[key][0])
        array = np.asarray(clusters[key])
        plt.scatter(x=array[:, 0], y=array[:, 1], marker='o')

    mid = np.asarray(mid)
    plt.scatter(x=mid[:, 0], y=mid[:, 1], marker='+', s=500)
    plt.show()


def main():
    data = pd.read_csv(r'Data/watermelon4.0.csv', index_col=0)
    X = np.array(data)

    clusters = k_means(X, n_class=4)
    plot_k_means(clusters)


if __name__ == '__main__':
    main()
