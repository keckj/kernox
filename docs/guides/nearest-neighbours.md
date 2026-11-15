# Nearest neighbours

kernox finds the K nearest neighbours of every point without computing the
full distance matrix.

## K nearest neighbours

```python
import torch
import kernox as kp

device = "cuda"

x, y = kp.Vi("x", 3), kp.Vj("y", 3)
d2 = x.sqdist(y)

knn = kp.compile(d2.argkmin(10, over="j"), dtype="float32")

X = torch.randn(2_000, 3, device=device)    # queries
Y = torch.randn(20_000, 3, device=device)   # database
idx = knn(x=X, y=Y)                          # (2000, 10) int64, nearest first
```

To get the distances too, use `kmin_argkmin`, which returns both in one pass:

```python
both = kp.compile(d2.kmin_argkmin(10, over="j"), dtype="float32")
dist2, idx = both(x=X, y=Y)                  # (2000, 10) float32 and int64
```

Neighbours are sorted by increasing distance; equal distances are ordered by
increasing index. `kmin` alone returns the distances only.

Some limits to know:

- The formula must be scalar (one distance per pair).
- On GPUs, `K` can go up to 1024. Larger values raise `UnsupportedError`.
- If `K` is larger than the number of candidates, the result is padded with
  index `-1` (and distance `+inf`).

## The single nearest neighbour

`argmin` (or `min_argmin` for distance and index together) is the K = 1 case:

```python
nearest = kp.compile(d2.argmin(over="j"), dtype="float32")(x=X, y=Y)   # (2000, 1)
```

## Other distances

Any scalar formula can be minimized. For example the L1 distance, or a cosine
dissimilarity:

```python
l1 = abs(x - y).sum()
cosine = 1 - x.normalize().dot(y.normalize())
idx_l1 = kp.compile(l1.argkmin(5, over="j"), dtype="float32")(x=X, y=Y)
```

To find the K largest scores instead, minimize the negated score:
`(-score).argkmin(K, over="j")`.

## Counting neighbours within a radius

A boolean formula reduced with `sum_over` counts the pairs where it holds:

```python
within = kp.compile((d2 < 0.1**2).sum_over("j"), dtype="float32")
counts = within(x=X, y=Y)                    # (2000, 1) int64
```

## Neighbours in the same set

To search a point cloud against itself, pass the same tensor twice. Each point
is then (barring exact duplicates) its own nearest neighbour, so ask for K + 1
neighbours and drop the first column. Here `knn` asks for 10, which gives 9:

```python
idx_self = knn(x=Y[:5_000], y=Y[:5_000])[:, 1:]
```

For very large clouds where only nearby clusters matter, combine this with
[block-sparse ranges](block-sparse.md).
