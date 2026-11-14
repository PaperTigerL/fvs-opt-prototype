# -*- coding: utf-8 -*-
"""实测各脚本耗时，写 results/runtime.txt。

为什么需要它：文档里的“3 秒 / 约 8 秒 / 合计 2.5 分钟”这类数字，靠人写就会漂——
本仓库就出现过报告写 3 秒、实测 4~8 秒的情况。把耗时也纳入“每个数字都能复现”的
范围，唯一办法是让脚本量、让文件存。

默认每个脚本跑 2 次，给出两次读数与较小值，用于暴露波动：
同一脚本在空闲机器与负载下可差 1.5 倍以上。

运行：python experiments/runtime.py [--repeat N]
输出：results/runtime.txt、results/runtime.csv
注意：会顺序跑一遍全部脚本，合计约 3–4 分钟（repeat=2）。
"""
import os
import sys
import csv
import time
import platform
import argparse
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPTS = [
    ('run_experiment.py', '主结果：三档可达率'),
    ('experiments/stitch_cost.py', '补边代价：边数/度数/边长/连通性'),
    ('experiments/ablation_long_edges.py', '长边消融'),
    ('experiments/beam_search.py', '固定预算 Recall@10（3 组种子）'),
    ('experiments/sensitivity.py', '入口/种子敏感性（12 组）'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeat', type=int, default=2, help='每个脚本测量次数，默认 2')
    args = ap.parse_args()
    repeat = max(1, args.repeat)

    rows = []
    for rel, what in SCRIPTS:
        runs = []
        status = 'ok'
        for _ in range(repeat):
            t0 = time.perf_counter()
            proc = subprocess.run([sys.executable, rel], cwd=HERE,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            runs.append(round(time.perf_counter() - t0, 2))
            if proc.returncode != 0:
                status = 'FAILED'
                sys.stderr.write(proc.stderr.decode('utf-8', 'ignore')[-400:])
                break
        rows.append((rel, what, runs, min(runs), status))
        print('%-34s %s  min=%.2fs  %s' % (rel, ' '.join('%.2fs' % r for r in runs), min(runs), status))

    out = os.path.join(HERE, 'results')
    os.makedirs(out, exist_ok=True)

    lines = [
        '# 各脚本实测耗时（由 experiments/runtime.py 生成，勿手改）',
        '',
        '环境：Python %s，%s' % (platform.python_version(), platform.platform()),
        '口径：含解释器启动与导入，冷启动；每行重复 %d 次。' % repeat,
        '',
        '波动说明：同一脚本在空闲机器与系统负载下可差 1.5 倍以上。',
        '本仓库实测 run_experiment.py 在空闲时约 4.5–4.7 秒，负载高时约 7.6 秒；',
        '因此文档写“约 5 秒（负载高时更久）”比写任何单点值更稳。',
        '',
        '%-40s %-22s %s' % ('script', 'runs (s)', 'what'),
    ]
    for rel, what, runs, best, status in rows:
        lines.append('%-40s %-22s %s%s' % (rel, ' / '.join('%.2f' % r for r in runs), what,
                                           '' if status == 'ok' else '  [FAILED]'))
    lines.append('')
    lines.append('%-40s %.2f' % ('合计（取每行较小值）', sum(r[3] for r in rows)))
    lines.append('')
    lines.append('文档引用本文件时写“见 results/runtime.txt”，不要另抄一个数字。')

    with open(os.path.join(out, 'runtime.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    with open(os.path.join(out, 'runtime.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['script', 'what'] + ['run_%d' % (i + 1) for i in range(repeat)] + ['min', 'status'])
        for rel, what, runs, best, status in rows:
            w.writerow([rel, what] + runs + [best, status])
        w.writerow(['TOTAL', '', '', '', round(sum(r[3] for r in rows), 2), ''])

    print('合计（较小值） %.2fs' % sum(r[3] for r in rows))
    print('已写入 results/runtime.txt、results/runtime.csv')
    return 0 if all(r[4] == 'ok' for r in rows) else 1


if __name__ == '__main__':
    sys.exit(main())
