# -*- coding: utf-8 -*-
"""标签诱导子图的连通性分析。

核心语义：BFS 只沿「两端都带同一标签」的边扩展，对应运行时前置过滤
（只访问合格节点）的实际行为。若某合格节点与入口之间没有一条全合格路径，
它在查询期就是不可达的——召回缺口由此产生。
"""


def components(nodes, adj):
    """返回 nodes 在 adj 上的连通分量列表（只沿 nodes 内部的边）。"""
    nodeset = set(nodes)
    seen = set()
    comps = []
    for s in nodes:
        if s in seen:
            continue
        stack = [s]
        seen.add(s)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v in nodeset and v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def reach_from(nodes, adj, entry):
    """从 entry 出发、只走合格节点的 BFS，返回可达节点占 nodes 的比例。"""
    nodeset = set(nodes)
    if entry not in nodeset:
        return 0.0
    seen = {entry}
    stack = [entry]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v in nodeset and v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) / float(len(nodes))


def nearest_of(nodes, points, target):
    """nodes 中离 target 空间位置最近的点（模拟「从全局入口进入该标签」）。"""
    best, bestd = None, 1e18
    for u in nodes:
        dx = points[u][0] - target[0]
        dy = points[u][1] - target[1]
        d = dx * dx + dy * dy
        if d < bestd:
            best, bestd = u, d
    return best


def analyze(nodes, adj, points, entry_pos):
    """一次完整测量：分量数、最大分量占比、从入口的可达率。"""
    comps = components(nodes, adj)
    entry = nearest_of(nodes, points, entry_pos)
    return {
        'n_nodes': len(nodes),
        'components': len(comps),
        'largest_frac': len(comps[0]) / float(len(nodes)) if comps else 0.0,
        'reach_rate': reach_from(nodes, adj, entry),
        'entry': entry,
    }
