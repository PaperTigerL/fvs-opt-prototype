# -*- coding: utf-8 -*-
"""主实验：三档 selectity × 补边前后，对比标签子图的连通性与可达率。

运行：python run_experiment.py
输出：results/connectivity.csv、results/edges_added.txt、控制台表格
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import dataset, graph, connectivity, patch, metrics


N_POINTS = 2000
K = 24          # kNN 度数，接近 Vamana 的 R=32 量级
N_LABELS = 40
ZIPF_A = 1.2
ENTRY_POS = (50.0, 50.0)      # 全局入口点：图的中心位置（模拟 medoid）


def main():
    print('生成数据：%d 点，%d 标签，k=%d' % (N_POINTS, N_LABELS, K))
    pts = dataset.gen_points(N_POINTS)
    labels = dataset.gen_labels(N_POINTS, N_LABELS, ZIPF_A)
    lidx = dataset.label_index(labels)

    adj = graph.build_knn(pts, K)
    deg = graph.degree_stats(adj)
    print('图：平均度 %.2f（min %d / max %d），无向' % (deg['mean'], deg['min'], deg['max']))

    buckets = dataset.pick_buckets(labels, N_POINTS)
    print('三档代表标签：' + ', '.join(
        '%s=标签%d(%.3f%%)' % (b, l, s * 100) for b, l, s in buckets))

    rows = []
    for name, l, s in buckets:
        m = connectivity.analyze(lidx[l], adj, pts, ENTRY_POS)
        m.update({'bucket': name, 'label': l, 'selectivity': s, 'stage': 'before'})
        rows.append(m)

    print('\n补边中（对所有 %d 个标签）…' % len(lidx))
    per_label, total_edges = patch.stitch_all(lidx, adj, pts)
    touched = sum(1 for v in per_label.values() if v > 0)
    print('补边完成：%d 个标签被缝合，共新增 %d 条边（原图 %d 条）' %
          (touched, total_edges, sum(len(a) for a in adj) // 2 - total_edges))

    for name, l, s in buckets:
        m = connectivity.analyze(lidx[l], adj, pts, ENTRY_POS)
        m.update({'bucket': name, 'label': l, 'selectivity': s, 'stage': 'after'})
        rows.append(m)

    print('\n' + metrics.fmt_table(rows))

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    metrics.write_csv(os.path.join(base, 'connectivity.csv'), rows)
    with open(os.path.join(base, 'edges_added.txt'), 'w', encoding='utf-8') as f:
        f.write('# 每个标签新增的边数（0 表示该标签本就连通）\n')
        for l in sorted(per_label):
            f.write('label_%-3d %d\n' % (l, per_label[l]))
        f.write('\n合计新增边 %d 条，涉及 %d 个标签\n' % (total_edges, touched))

    print('\n结果已写入 results/connectivity.csv 与 results/edges_added.txt')

    # 结论摘要
    d = {}
    for r in rows:
        d[(r['bucket'], r['stage'])] = r
    print('\n=== 结论 ===')
    for name, l, s in buckets:
        b = d[(name, 'before')]
        a = d[(name, 'after')]
        print('%s（sel %.3f%%）：可达率 %.1f%% → %.1f%%（+%.1f pp），分量 %d → %d' % (
            name, s * 100, b['reach_rate'] * 100, a['reach_rate'] * 100,
            (a['reach_rate'] - b['reach_rate']) * 100, b['components'], a['components']))


if __name__ == '__main__':
    main()
