# -*- coding: utf-8 -*-
"""消融：长边对标签子图连通性的影响（README §3 引用）。

整张图的连通性本身不说明问题：kNN 双向化后，簇边界上的点会把相邻簇连起来，
所以即便没有长边，全图也常常是单一连通分量。真正受影响的是**标签诱导子图**——
随机长边给了同一标签跨簇跳转的机会，缺了它，大标签的子图也会被切碎。

运行：python experiments/ablation_long_edges.py
输出：results/ablation_long_edges.csv
"""
import os
import sys
import csv
import statistics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from src import dataset, graph as g_mod, connectivity as c_mod

N_POINTS, K, N_LABELS, ZIPF_A = 2000, 24, 40, 1.2
ENTRY = (50.0, 50.0)


def main():
    pts = dataset.gen_points(N_POINTS)
    labels = dataset.gen_labels(N_POINTS, N_LABELS, ZIPF_A)
    lidx = dataset.label_index(labels)
    buckets = dataset.pick_buckets(labels, N_POINTS)

    rows = []
    for n_long in (0, 4):
        adj = g_mod.build_knn(pts, K, n_long, 23)
        degs = [len(a) for a in adj]
        whole = c_mod.components(list(range(N_POINTS)), adj)
        for name, l, s in buckets:
            nodes = lidx[l]
            comps = c_mod.components(nodes, adj)
            entry = c_mod.nearest_of(nodes, pts, ENTRY)
            ec = [c for c in comps if entry in c][0]
            rows.append({
                'n_long': n_long, 'bucket': name, 'label': l,
                'selectivity': round(s, 5), 'n_nodes': len(nodes),
                'mean_degree': round(sum(degs) / float(N_POINTS), 2),
                'graph_components': len(whole),
                'label_components': len(comps),
                'largest_frac': round(len(comps[0]) / float(len(nodes)), 4),
                'reach': round(len(ec) / float(len(nodes)), 4),
            })

    out = os.path.join(HERE, 'results')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'ablation_long_edges.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print('%-8s %-7s %-9s %-8s %-8s %s' % ('n_long', 'bucket', 'sel', 'labelc', 'graphc', 'reach'))
    for r in rows:
        print('%-8d %-7s %-9s %-8d %-8d %.1f%%' % (
            r['n_long'], r['bucket'], '%.3f%%' % (r['selectivity'] * 100),
            r['label_components'], r['graph_components'], r['reach'] * 100))
    print('\n已写入 results/ablation_long_edges.csv')


if __name__ == '__main__':
    main()
