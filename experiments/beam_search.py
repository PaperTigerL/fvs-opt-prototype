# -*- coding: utf-8 -*-
"""预算受限的过滤搜索：补边究竟救回了多少 Recall@10（README §8 引用）。

为什么需要这个实验：主实验的 reach_rate 在补边后必然等于 100%——
补边把诱导子图连成一片，可达性就是构造出来的结果，它证明不了搜索质量。
真正的检验是：在**固定搜索预算**下，补边前后的 Recall@10 各是多少。

搜索模拟 Vamana 式的 best-first 扩展，只允许访问合格节点（前置过滤语义）：
  1. 入口取「离图中心最近的合格点」，与主实验一致；
  2. 反复弹出候选集中离查询最近、且尚未展开的节点，把它的合格邻居加入候选集；
  3. 预算 = 展开节点次数，展开 budget 次即停；
  4. 结果集 = 已访问节点中离查询最近的 K 个，与真值（全体合格点中最近 K 个）比对。

运行：python experiments/beam_search.py
输出：results/beam_recall.csv
"""
import os
import sys
import csv
import random
import statistics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from src import dataset, graph as g_mod, patch, connectivity as c_mod

N_POINTS, K, N_LABELS, ZIPF_A = 2000, 24, 40, 1.2
ENTRY = (50.0, 50.0)
TOPK = 10
N_QUERIES = 40
BUDGETS = (10, 20, 40, 80, 160)


def d2(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy


def beam_search(adj, pts, qualified, entry, query, budget):
    """只走合格节点的 best-first 扩展，返回已访问节点集合。"""
    visited = {entry}
    frontier = {entry: d2(pts[entry], query)}
    expanded = 0
    while frontier and expanded < budget:
        u = min(frontier, key=frontier.get)
        del frontier[u]
        expanded += 1
        for v in adj[u]:
            if v in visited or v not in qualified:
                continue
            visited.add(v)
            frontier[v] = d2(pts[v], query)
    return visited


def make_queries(pts, lidx, buckets, seed=101):
    """查询点 = 随机合格点加抖动；真值 = 全体合格点中最近的 K 个。"""
    rng = random.Random(seed)
    out = {}
    for name, l, s in buckets:
        nodes = lidx[l]
        lst = []
        for _ in range(N_QUERIES):
            src = rng.choice(nodes)
            q = (pts[src][0] + rng.gauss(0, 1.5), pts[src][1] + rng.gauss(0, 1.5))
            gt = sorted(nodes, key=lambda v: d2(pts[v], q))[:TOPK]
            lst.append((q, gt))
        out[name] = lst
    return out


def run_stage(adj, pts, lidx, buckets, queries, stage, rows):
    for name, l, s in buckets:
        nodes = lidx[l]
        qset = set(nodes)
        entry = c_mod.nearest_of(nodes, pts, ENTRY)
        for b in BUDGETS:
            rs = []
            for q, gt in queries[name]:
                visited = beam_search(adj, pts, qset, entry, q, b)
                found = sorted(visited, key=lambda v: d2(pts[v], q))[:TOPK]
                rs.append(len(set(found) & set(gt)) / float(len(gt)))
            rows.append({'bucket': name, 'selectivity': round(s, 5), 'n_nodes': len(nodes),
                         'stage': stage, 'budget': b, 'recall_at_10': round(statistics.mean(rs), 4)})


def main():
    pts = dataset.gen_points(N_POINTS)
    labels = dataset.gen_labels(N_POINTS, N_LABELS, ZIPF_A)
    lidx = dataset.label_index(labels)
    buckets = dataset.pick_buckets(labels, N_POINTS)
    queries = make_queries(pts, lidx, buckets)

    adj = g_mod.build_knn(pts, K)
    rows = []
    run_stage(adj, pts, lidx, buckets, queries, 'before', rows)
    patch.stitch_all(lidx, adj, pts)
    run_stage(adj, pts, lidx, buckets, queries, 'after', rows)

    out = os.path.join(HERE, 'results')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'beam_recall.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print('%-7s %-9s %-7s %s' % ('bucket', 'sel', 'stage',
                                 '  '.join('b=%d' % b for b in BUDGETS)))
    for name, l, s in buckets:
        for stage in ('before', 'after'):
            cells = [r['recall_at_10'] for r in rows
                     if r['bucket'] == name and r['stage'] == stage]
            print('%-7s %-9s %-7s %s' % (name, '%.3f%%' % (s * 100), stage,
                                         '  '.join('%5.1f%%' % (c * 100) for c in cells)))
    print('\n已写入 results/beam_recall.csv')


if __name__ == '__main__':
    main()
