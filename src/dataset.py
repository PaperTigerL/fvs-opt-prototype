# -*- coding: utf-8 -*-
"""合成数据集：几何聚类点集 + Zipf 标签。

关键设定：点坐标与标签**独立生成**。这正是真实场景的困难来源——
图索引按坐标的邻近关系连边，谓词按标签筛选，两者不共享生成过程，
因此同一标签的节点在图上会被切成多个碎片。
"""
import math
import random


def gen_points(n, clusters=24, spread=6.0, field=100.0, seed=7):
    """在 100x100 平面上生成 n 个点，来自若干高斯簇（模拟向量的几何聚类）。"""
    rng = random.Random(seed)
    centers = [(rng.uniform(0, field), rng.uniform(0, field)) for _ in range(clusters)]
    pts = []
    for _ in range(n):
        cx, cy = rng.choice(centers)
        pts.append((rng.gauss(cx, spread), rng.gauss(cy, spread)))
    return pts


def gen_labels(n, n_labels=40, zipf_a=1.2, seed=11):
    """Zipf 分布的标签：少数标签覆盖大多数点，长尾标签很稀少。

    每个点带 1~3 个标签（多数为 1 个），模拟多标签属性。
    """
    rng = random.Random(seed)
    raw = [1.0 / (i + 1) ** zipf_a for i in range(n_labels)]
    total = sum(raw)
    cum, acc = [], 0.0
    for w in raw:
        acc += w / total
        cum.append(acc)

    def pick():
        r = rng.random()
        for idx, c in enumerate(cum):
            if r <= c:
                return idx
        return n_labels - 1

    labels = []
    for _ in range(n):
        k = rng.choices([1, 2, 3], weights=[0.75, 0.20, 0.05])[0]
        labels.append(sorted({pick() for _ in range(k)}))
    return labels


def label_index(labels):
    """标签 -> 节点列表。"""
    idx = {}
    for i, ls in enumerate(labels):
        for l in ls:
            idx.setdefault(l, []).append(i)
    return idx


def pick_buckets(labels, n_points, targets=(0.005, 0.03, 0.12)):
    """按标签规模挑出接近三档 selectivity 的代表标签。

    返回 [(档位名, 标签, 实际 selectivity)]，实际值写入结果，避免把名义值当实测值。
    """
    idx = label_index(labels)
    out = []
    used = set()
    for tgt, name in zip(targets, ('Low', 'Medium', 'High')):
        best, best_err = None, 1e9
        for l, nodes in idx.items():
            if l in used:
                continue
            s = len(nodes) / float(n_points)
            err = abs(math.log((s + 1e-9) / tgt))
            if err < best_err:
                best, best_err = (l, s), err
        if best:
            used.add(best[0])
            out.append((name, best[0], best[1]))
    return out
