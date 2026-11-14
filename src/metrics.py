# -*- coding: utf-8 -*-
"""指标汇总与 CSV 输出。"""
import csv
import os


FIELDS = ['bucket', 'label', 'selectivity', 'stage',
          'n_nodes', 'components', 'largest_frac', 'reach_rate']


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in FIELDS})


def fmt_table(rows):
    """控制台表格：每档标签一行，补边前后对照。"""
    out = []
    head = ('%-7s %-6s %-8s %-6s %6s %6s %9s %9s' %
            ('bucket', 'label', 'sel', 'stage', 'nodes', 'comps', 'largest', 'reach'))
    out.append(head)
    out.append('-' * len(head))
    for r in rows:
        out.append('%-7s %-6s %-8s %-6s %6d %6d %8.1f%% %8.1f%%' % (
            r['bucket'], r['label'], '%.3f%%' % (r['selectivity'] * 100), r['stage'],
            r['n_nodes'], r['components'],
            r['largest_frac'] * 100, r['reach_rate'] * 100))
    return '\n'.join(out)
