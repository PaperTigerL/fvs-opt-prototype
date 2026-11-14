# -*- coding: utf-8 -*-
"""kNN 图构建：每个点连到欧氏距离最近的 k 个点。

只做最近邻、不做 Vamana 的剪枝，目的是隔离变量——本实验关心的是
「标签子图是否连通」，而不是图质量本身。
"""


def _d2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def build_knn(points, k=24, n_long=4, seed=23):
    """kNN 图 + 每点若干条随机长边。

    长边不是随手加的：Vamana 的 α 剪枝会特意保留一部分「远距离但方向不同」的边，
    使图能跨簇跳转。缺了这层，kNN 图在簇间完全不连通，会把大标签的子图也切碎，
    与真实索引的行为不符。
    """
    import random
    n = len(points)
    adj = [[] for _ in range(n)]
    for i in range(n):
        pi = points[i]
        dists = []
        for j in range(n):
            if i != j:
                dists.append((_d2(pi, points[j]), j))
        dists.sort()
        adj[i] = [j for _, j in dists[:k]]

    if n_long:
        rng = random.Random(seed)
        for i in range(n):
            for _ in range(n_long):
                j = rng.randrange(n)
                if j != i:
                    adj[i].append(j)

    # 双向化：kNN 是有向的，不补反向边会出现「单行道」，
    # 与真实图索引（对称边）不一致，会高估碎片化。
    und = [set(a) for a in adj]
    for i, ns in enumerate(adj):
        for j in ns:
            und[j].add(i)
    return [sorted(s) for s in und]


def degree_stats(adj):
    degs = [len(a) for a in adj]
    n = len(degs)
    return {
        'min': min(degs),
        'max': max(degs),
        'mean': sum(degs) / float(n),
    }
