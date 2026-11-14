# -*- coding: utf-8 -*-
"""补边的结构代价：边数、度数变化、边长、连通性（README §5/§6、报告 §5.8 引用）。

README 里那几句「补边长度中位数 11.3、最长 57.8」「平均度只升 0.62、最大度不变」
「40 个标签全部连通、跨标签边 0 条」都由本脚本产出，避免报告里出现无法复现的数字。

运行：python experiments/stitch_cost.py
输出：results/stitch_cost.csv、results/stitch_cost.txt
"""
import os
import sys
import csv
import math
import statistics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from src import dataset, graph as g_mod, patch, connectivity as c_mod

N_POINTS, K, N_LABELS, ZIPF_A = 2000, 24, 40, 1.2


def dist(p, q):
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


def main():
    pts = dataset.gen_points(N_POINTS)
    labels = dataset.gen_labels(N_POINTS, N_LABELS, ZIPF_A)
    lidx = dataset.label_index(labels)

    adj = g_mod.build_knn(pts, K)
    deg_before = [len(a) for a in adj]
    orig = [set(a) for a in adj]
    edges_before = sum(deg_before) // 2
    klen = [dist(pts[i], pts[j]) for i, ns in enumerate(adj) for j in ns if j > i]

    per_label, added = patch.stitch_all(lidx, adj, pts)

    deg_after = [len(a) for a in adj]
    edges_after = sum(deg_after) // 2
    new_edges = [(i, j) for i, ns in enumerate(adj) for j in ns
                 if j > i and j not in orig[i]]
    slen = [dist(pts[i], pts[j]) for i, j in new_edges]

    cross = sum(1 for i, j in new_edges if not set(labels[i]) & set(labels[j]))
    not_connected = [l for l, nodes in lidx.items() if len(c_mod.components(nodes, adj)) > 1]
    touched = sum(1 for v in per_label.values() if v > 0)
    added_per_node = [deg_after[i] - deg_before[i] for i in range(N_POINTS)]

    rows = [
        ('edges_before', edges_before),
        ('edges_after', edges_after),
        ('edges_added', added),
        ('pct_added_of_original', round(added / float(edges_before) * 100, 4)),
        ('degree_mean_before', round(statistics.mean(deg_before), 3)),
        ('degree_mean_after', round(statistics.mean(deg_after), 3)),
        ('degree_max_before', max(deg_before)),
        ('degree_max_after', max(deg_after)),
        ('max_edges_added_per_node', max(added_per_node)),
        ('edge_len_knn_median', round(statistics.median(klen), 2)),
        ('edge_len_knn_mean', round(statistics.mean(klen), 2)),
        ('edge_len_stitched_median', round(statistics.median(slen), 2)),
        ('edge_len_stitched_mean', round(statistics.mean(slen), 2)),
        ('edge_len_stitched_max', round(max(slen), 2)),
        ('edge_len_ratio_median', round(statistics.median(slen) / statistics.median(klen), 2)),
        ('labels_total', len(lidx)),
        ('labels_touched', touched),
        ('labels_not_connected_after', len(not_connected)),
        ('stitched_edges_crossing_labels', cross),
    ]

    out = os.path.join(HERE, 'results')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'stitch_cost.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerows(rows)

    lines = ['# 补边的结构代价', '']
    lines += ['%-32s %s' % (k, v) for k, v in rows]
    txt = os.path.join(out, 'stitch_cost.txt')
    with open(txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('\n已写入 results/stitch_cost.csv、results/stitch_cost.txt')


if __name__ == '__main__':
    main()
