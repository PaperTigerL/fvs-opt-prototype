# -*- coding: utf-8 -*-
"""稳健性检查：入口位置敏感性与随机种子敏感性。

背景：reach_rate 的定义是「从入口出发的 BFS 能到达的合格节点占比」，
它等于**入口所在连通分量的大小占比**。因此这个数依赖于入口落在哪个碎片，
单次测量只是一个抽样。本脚本量化这一点，避免把单次抽样当成档位属性。

运行：python experiments/sensitivity.py
输出：results/sensitivity.csv、results/sensitivity_summary.txt
"""
import os
import sys
import csv
import statistics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from src import dataset, graph as g_mod, connectivity as c_mod

N_POINTS, K, N_LABELS, ZIPF_A = 2000, 24, 40, 1.2
ENTRY = (50.0, 50.0)          # 与主实验一致的入口：图中心位置


def reach_of_entry(nodes, adj, entry):
    comps = c_mod.components(nodes, adj)
    comp = [c for c in comps if entry in c][0]
    return len(comp) / float(len(nodes)), len(comps)


def entry_sweep(nodes, adj, pts):
    """遍历所有可能入口，给出 reach 的分布。"""
    comps = c_mod.components(nodes, adj)
    n = len(nodes)
    vals = []
    for c in comps:
        vals += [len(c) / float(n)] * len(c)
    return {
        'entry_mean': sum(vals) / len(vals),
        'entry_min': min(vals),
        'entry_median': statistics.median(vals),
        'entry_max': max(vals),
    }


def main():
    rows = []
    for dseed in (7, 11, 13):
        pts = dataset.gen_points(N_POINTS, seed=dseed)
        labels = dataset.gen_labels(N_POINTS, N_LABELS, ZIPF_A, seed=dseed + 100)
        lidx = dataset.label_index(labels)
        buckets = dataset.pick_buckets(labels, N_POINTS)
        for gseed in (23, 29, 31, 37):
            adj = g_mod.build_knn(pts, K, 4, gseed)
            for name, l, s in buckets:
                nodes = lidx[l]
                entry = c_mod.nearest_of(nodes, pts, ENTRY)
                reach, ncomp = reach_of_entry(nodes, adj, entry)
                sw = entry_sweep(nodes, adj, pts)
                rows.append({
                    'dataset_seed': dseed, 'graph_seed': gseed, 'bucket': name,
                    'label': l, 'selectivity': round(s, 5), 'n_nodes': len(nodes),
                    'components': ncomp,
                    'reach_at_center_entry': round(reach, 4),
                    'reach_entry_mean': round(sw['entry_mean'], 4),
                    'reach_entry_min': round(sw['entry_min'], 4),
                    'reach_entry_median': round(sw['entry_median'], 4),
                    'reach_entry_max': round(sw['entry_max'], 4),
                })

    out = os.path.join(HERE, 'results')
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'sensitivity.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = ['# 稳健性汇总：%d 行（数据集种子 x 图种子 x 档位）' % len(rows), '']
    for b in ('Low', 'Medium', 'High'):
        sub = [r for r in rows if r['bucket'] == b]
        rs = [r['reach_at_center_entry'] for r in sub]
        em = [r['reach_entry_mean'] for r in sub]
        lines.append('%s: reach(中心入口) min %.1f%% / median %.1f%% / max %.1f%% / mean %.1f%%;'
                     ' 入口平均 %.1f%%; 分量数 %d-%d'
                     % (b, min(rs) * 100, statistics.median(rs) * 100, max(rs) * 100,
                        statistics.mean(rs) * 100, statistics.mean(em) * 100,
                        min(r['components'] for r in sub), max(r['components'] for r in sub)))
    txt = os.path.join(out, 'sensitivity_summary.txt')
    with open(txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('\n已写入 results/sensitivity.csv、results/sensitivity_summary.txt')


if __name__ == '__main__':
    main()
