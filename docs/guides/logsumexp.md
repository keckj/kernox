# Log-sum-exp, softmax and Sinkhorn

Log-sum-exp reductions are the stable way to sum exponentials whose exponents
can be large or very negative. They appear in soft-min operators, attention,
mixture models and entropic optimal transport.

## Log-sum-exp

`f.logsumexp(over="j")` computes `log Σ_j exp f(i, j)` for each component,
in a single pass and without overflow:

```python
import torch
import kernox as kp

device = "cuda"

x, y = kp.Vi("x", 2), kp.Vj("y", 2)
eps = kp.Param("eps")
C = x.sqdist(y) / 2                                  # cost of moving x_i to y_j

softmin = kp.compile((-C / eps).logsumexp(over="j"), dtype="float32")

X = torch.randn(3_000, 2, device=device)
Y = torch.randn(4_000, 2, device=device) + 1.0
L = softmin(x=X, y=Y, eps=0.01)                      # (3000, 1)
```

Even with `eps = 0.01`, where `exp(-C / eps)` underflows to zero in float32,
the result is accurate, because kernox shifts by the running maximum.

## Weighted log-sum-exp

With a scalar score `f` and positive weights `g`,
`f.logsumexp(over="j", weight=g)` computes `log Σ_j exp(f) g_j`:

```python
b = kp.Vj("b", 1)
weighted = kp.compile((-C / eps).logsumexp(over="j", weight=b), dtype="float32")
B = torch.rand(4_000, 1, device=device)
Lw = weighted(x=X, y=Y, b=B, eps=0.01)               # (3000, 1)
```

Negative weights give NaN, like `log` of a negative number.

## Softmax-weighted sums

`f.sumsoftmaxweight(g, over="j")` is the softmax of `f` over `j`, applied to the
vectors `g`: `Σ_j exp(f) g / Σ_j exp(f)`. This is attention with a formula as
the score:

```python
q, k = kp.Vi("q", 8), kp.Vj("k", 8)
v = kp.Vj("v", 4)
attention = kp.compile((q.dot(k) / 8**0.5).sumsoftmaxweight(v, over="j"), dtype="float32")

Q = torch.randn(1_000, 8, device=device)
K = torch.randn(2_000, 8, device=device)
V = torch.randn(2_000, 4, device=device)
out = attention(q=Q, k=K, v=V)                       # (1000, 4)
```

## Sinkhorn iterations

The log-domain Sinkhorn algorithm for entropic optimal transport between
uniform measures alternates two soft-min updates of the dual potentials `f`
and `g`. Each update is one log-sum-exp, over `j` for `f` and over `i` for `g`:

```python
f, g = kp.Vi("f", 1), kp.Vj("g", 1)

update_f = kp.compile(((g - C) / eps).logsumexp(over="j"), dtype="float32")   # (M, 1)
update_g = kp.compile(((f - C) / eps).logsumexp(over="i"), dtype="float32")   # (N, 1)

M, N = X.shape[0], Y.shape[0]
e = 0.05
F = torch.zeros(M, 1, device=device)
G = torch.zeros(N, 1, device=device)
for _ in range(50):
    F = -e * (update_f(x=X, y=Y, g=G, eps=e) - torch.log(torch.tensor(float(N))))
    G = -e * (update_g(x=X, y=Y, f=F, eps=e) - torch.log(torch.tensor(float(M))))

cost = F.mean() + G.mean()                           # entropic transport cost
```

Both reductions compile once; each iteration only launches two kernels. Since
`eps` is a `Param`, decreasing it along the iterations (epsilon scaling)
needs no recompilation.

For more speed, `fast_math=True` lets the CUDA backend use approximate `exp`
and `log` in float32: see [fast_math](fast-math.md).

## Special values

- A score of `-inf` contributes nothing.
- With no terms (N = 0) or only `-inf` scores, `logsumexp` returns `-inf` and
  `sumsoftmaxweight` returns NaN.
- A `+inf` score gives `+inf` for `logsumexp`, and NaN for `sumsoftmaxweight`.

## Gradients

All three reductions support first-order gradients through `torch.autograd`,
computed as fused reductions as well. See [Gradients](gradients.md).
