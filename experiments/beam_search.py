# -*- coding: utf-8 -*-
"""预算受限的过滤搜索：补边究竟救回了多少 Recall@10（README §8 引用）。

为什么需要这个实验：主实验的 reach_rate 在补边后必然等于 100%——
补边把诱导子图连成一片，可达性就是构造出来的结果，它证明不了搜索质量。
真正的检验是：在**固定搜索预算**下，补边前后的 Recall@10 各是多少。

搜索模拟 Vamana 式的 best-first 扩展，只允许访问合格节点（前置过滤语义）：
  1. 入口取「离图中心最近的合格点」或「随机合格点」，两种策略分别报告；
  2. 反复弹出候选集中离查询最近、且尚未展开的节点，把它的合格邻居加入候选集；
  3. 预算 = 展开节点次数，展开 budget 次即停；
  4. 结果集 = 已访问节点中离查询最近的 K 个，与真值（全体合格点中最近 K 个）比对。

**重复设置**：3 组独立种子（数据集 / 标签 / 查询各自重新生成），每档每预算得 3 个读数，
报告 中位数与 min–max。单次运行的读数不足以支撑结论——§7 已说明 reach 依赖入口落点。

运行：python experiments/beam_search.py
输出：results/beam_recall.csv（逐次）、results/beam_recall_summary.csv（汇总）
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
SEEDS = ((7, 11), (17, 117), (27, 127))   # (点种子, 标签种子)；第一组与主实验一致
N_RANDOM_ENTRIES = 5          # 随机入口策略下每档取的入口数


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


def make_queries(pts, lidx, buckets, seed):
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


def run_stage(adj, pts, lidx, buckets, queries, stage, seed, rows):
    for name, l, s in buckets:
        nodes = lidx[l]
        qset = set(nodes)
        entry = c_mod.nearest_of(nodes, pts, ENTRY)

        rng = random.Random(seed * 31 + 5)
        rand_entries = [rng.choice(nodes) for _ in range(N_RANDOM_ENTRIES)]

        for b in BUDGETS:
            # 固定中心入口
            rs = []
            for q, gt in queries[name]:
                visited = beam_search(adj, pts, qset, entry, q, b)
                found = sorted(visited, key=lambda v: d2(pts[v], q))[:TOPK]
                rs.append(len(set(found) & set(gt)) / float(len(gt)))
            # 随机入口：每个入口跑同一批查询，取平均
            rs_rand = []
            for ent in rand_entries:
                for q, gt in queries[name]:
                    visited = beam_search(adj, pts, qset, ent, q, b)
                    found = sorted(visited, key=lambda v: d2(pts[v], q))[:TOPK]
                    rs_rand.append(len(set(found) & set(gt)) / float(len(gt)))

            rows.append({
                'seed': seed, 'bucket': name, 'selectivity': round(s, 5),
                'n_nodes': len(nodes), 'stage': stage, 'budget': b,
                'recall_center': round(statistics.mean(rs), 4),
                'recall_random': round(statistics.mean(rs_rand), 4),
            })


def main():
    all_rows = []
    for seed, lseed in SEEDS:
        pts = dataset.gen_points(N_POINTS, seed=seed)
        labels = dataset.gen_labels(N_POINTS, N_LABELS, ZIPF_A, seed=lseed)
        lidx = dataset.label_index(labels)
        buckets = dataset.pick_buckets(labels, N_POINTS)
        queries = make_queries(pts, lidx, buckets, seed + 200)

        adj = g_mod.build_knn(pts, K)
        run_stage(adj, pts, lidx, buckets, queries, 'before', seed, all_rows)
        patch.stitch_all(lidx, adj, pts)
        run_stage(adj, pts, lidx, buckets, queries, 'after', seed, all_rows)

    out = os.path.join(HERE, 'results')
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(out, 'beam_recall.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    # 汇总：每 (bucket, stage, budget) 跨种子的中位数与 min-max
    summary = []
    keys = sorted({(r['bucket'], r['stage'], r['budget']) for r in all_rows},
                  key=lambda x: (x[0], x[1], x[2]))
    for bucket, stage, budget in keys:
        sel = [r for r in all_rows
               if r['bucket'] == bucket and r['stage'] == stage and r['budget'] == budget]
        vals = [r['recall_center'] for r in sel]
        vals_r = [r['recall_random'] for r in sel]
        summary.append({
            'bucket': bucket, 'stage': stage, 'budget': budget,
            'n_nodes': sel[0]['n_nodes'], 'selectivity': sel[0]['selectivity'],
            'center_median': round(statistics.median(vals), 4),
            'center_min': round(min(vals), 4), 'center_max': round(max(vals), 4),
            'random_median': round(statistics.median(vals_r), 4),
            'runs': len(vals),
        })

    with open(os.path.join(out, 'beam_recall_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    for name in dict.fromkeys(s['bucket'] for s in summary):
        sub = [s for s in summary if s['bucket'] == name]
        n_nodes = sub[0]['n_nodes']
        sel = sub[0]['selectivity']
        print('\n%s  标签节点数 %d  (sel %.3f%%)' % (name, n_nodes, sel * 100))
        print('  %-8s %-10s %s' % ('stage', 'entry', '  '.join('b=%d' % b for b in BUDGETS)))
        for stage in ('before', 'after'):
            for key, label in (('center_median', '中心'), ('random_median', '随机')):
                vals = [s[key] for s in sub if s['stage'] == stage]
                print('  %-8s %-10s %s' % (stage, label,
                                           '  '.join('%5.1f%%' % (v * 100) for v in vals)))
        # 误差范围（中心入口）
        for stage in ('before', 'after'):
            lo = [s['center_min'] for s in sub if s['stage'] == stage]
            hi = [s['center_max'] for s in sub if s['stage'] == stage]
            print('  %-8s 范围       %s' % (stage,
                  '  '.join('%4.0f-%4.0f' % (l * 100, h * 100) for l, h in zip(lo, hi))))

    print('\n已写入 results/beam_recall.csv 与 results/beam_recall_summary.csv')


if __name__ == '__main__':
    main()
