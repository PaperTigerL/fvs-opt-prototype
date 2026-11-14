# -*- coding: utf-8 -*-
"""构建期同标签补边（label-aware edge stitching）。

思路与 FilteredVamana 的「建图时让同标签节点更容易相邻」同源，
但实现更轻：不改建图算法，只在建完图后对每个标签做一次离线缝合。

对标签 L：
  1. 求 L 诱导子图的连通分量；
  2. 取最大分量为「主分量」；
  3. 每次挑一对「非主分量节点 a、主分量节点 b」使距离最小，加边 a—b，
     把 a 所在分量并入主分量；
  4. 直到 L 只剩一个分量，或达到该标签的补边预算。

选最近的一对，是为了让补的边尽量短，不破坏原有几何局部性。
"""
from .connectivity import components


def _d2(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy


def stitch_label(nodes, adj, points, budget=None):
    """对单个标签补边，原地修改 adj，返回新增边列表 [(a, b), ...]。"""
    added = []
    budget = budget if budget is not None else len(nodes)
    while len(added) < budget:
        comps = components(nodes, adj)
        if len(comps) <= 1:
            break
        main = comps[0]
        rest = comps[1:]

        best = None
        bestd = 1e18
        for comp in rest:
            for a in comp:
                pa = points[a]
                for b in main:
                    d = _d2(pa, points[b])
                    if d < bestd:
                        bestd = d
                        best = (a, b)
        if best is None:
            break
        a, b = best
        adj[a].append(b)
        adj[b].append(a)
        added.append((a, b))
    return added


def stitch_all(label_index, adj, points, budget_per_label=None):
    """对所有标签补边。返回 {label: 新增边数} 与总边数。"""
    per_label = {}
    total = 0
    for l, nodes in sorted(label_index.items()):
        if len(nodes) < 2:
            per_label[l] = 0
            continue
        added = stitch_label(nodes, adj, points, budget_per_label)
        per_label[l] = len(added)
        total += len(added)
    return per_label, total
