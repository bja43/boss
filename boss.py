from numpy import zeros
from numpy.random import default_rng

from gst import GST
from scores import BIC

import numpy as np

from dao import er_dag, corr, simulate


def boss(order, gsts, rng=default_rng()):
    variables = [v for v in order]
    while True:
        improved = False
        rng.shuffle(variables)
        for v in variables:
            improved |= better_mutation(v, order, gsts)
        if not improved: break
    return order


def reversed_enumerate(iter, j):
    for w in reversed(iter):
        yield j, w
        j -= 1


def better_mutation(v, order, gsts):
    i = order.index(v)
    p = len(order)
    scores = zeros(p + 1)

    prefix = []
    score = 0
    for j, w in enumerate(order):
        scores[j] = gsts[v].trace(prefix) + score
        if v != w:
            score += gsts[w].trace(prefix)
            prefix.append(w)

    scores[p] = gsts[v].trace(prefix) + score
    best = p

    prefix.append(v)
    score = 0
    for j, w in reversed_enumerate(order, p - 1):
        if v != w:
            prefix.remove(w)
            score += gsts[w].trace(prefix)
        scores[j] += score
        if scores[j] > scores[best]: best = j
        
    if scores[i] + 1e-6 > scores[best]: return False
    order.remove(v)
    order.insert(best - int(best > i), v)

    return True



p = 100
n = 1000

g = er_dag(p, ad=10)
_, B, O = corr(g)
X = simulate(B, O, n)

score = BIC(X, pd=2)

order = [v for v in range(p)]
gsts = [GST(v, score) for v in order]

default_rng().shuffle(order)
boss(order, gsts)

parents = {v: [] for v in order}
for i, v in enumerate(order): gsts[v].trace(order[:i], parents[v])

for v in parents:
    print(v)
    print("True:", sorted(np.where(g[v])[0]))
    print("Est: ", sorted(parents[v]))
    print()