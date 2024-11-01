# BOSS
A Python implementation of the best order score search causal discovery algorithm.

```python
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
```
