# Block-sparse reductions

When the formula is negligible for most pairs (a Gaussian kernel with a small
bandwidth, neighbours within a radius), you can skip whole blocks of pairs.
Sort the points into clusters, decide which pairs of clusters interact, and
kernox only visits those. This is the `ranges` mechanism of PyKeOps.

## Step by step

```python
import torch
import kernox as kp

device = "cuda"

x, y, b = kp.Vi("x", 2), kp.Vj("y", 2), kp.Vj("b", 1)
sigma = 0.05
gauss = kp.compile((kp.exp(-x.sqdist(y) / (2 * sigma**2)) * b).sum_over("j"), dtype="float32")

X = torch.rand(4_000, 2, device=device)
Y = torch.rand(5_000, 2, device=device)
B = torch.randn(5_000, 1, device=device)
```

**1. Label the points with a cluster.** Here a simple 10 × 10 grid on the unit
square; any clustering works (k-means, octree, ...).

```python
def grid_labels(P, n=10):
    cell = (P * n).long().clamp(0, n - 1)
    return cell[:, 0] * n + cell[:, 1]

labels_x, labels_y = grid_labels(X), grid_labels(Y)
```

**2. Sort the points by label**, and keep the permutation to undo it later.
`ranges_from_labels` needs sorted labels.

```python
labels_x, perm_x = torch.sort(labels_x)
labels_y, perm_y = torch.sort(labels_y)
Xs, Ys, Bs = X[perm_x], Y[perm_y], B[perm_y]
```

**3. Choose which cluster pairs interact.** `keep[c, d]` is `True` when the
points of cluster `c` must see those of cluster `d`. Here: cells closer than
a few `sigma`.

```python
n = 10
centers = torch.stack(torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij"), -1)
centers = (centers.reshape(-1, 2).float() + 0.5).to(device) / n
keep = torch.cdist(centers, centers) < 1 / n + 6 * sigma      # (100, 100) bool
```

**4. Build the ranges and reduce.**

```python
r = kp.ranges_from_labels(labels_x, labels_y, keep)
out_sorted = gauss.with_ranges(r)(x=Xs, y=Ys, b=Bs)          # (4000, 1)

out = torch.empty_like(out_sorted)
out[perm_x] = out_sorted                                     # back to the order of X
```

The result matches the dense reduction up to the neglected terms, here smaller
than `exp(-18)` relative to the kernel peak.

## Rules

- For a row `i`, the reduction runs over the `j` of all clusters kept for the
  cluster of `i`. A row whose cluster interacts with nothing gets the neutral
  value of the reduction (0 for a sum, `+inf` for a min, `-1` for an index).
- Returned indices refer to the sorted `Y` (`Ys` above), not to the original
  order. Map them back with `perm_y[idx]`.
- Ties still go to the smallest index.
- `with_ranges(r, validate=True)` checks that the ranges are in bounds and do
  not overlap. It costs one synchronization with the host per call, so use it
  while developing.

## Gradients and reductions over `i`

`ranges_from_labels` builds the ranges in both directions, so gradients with
respect to `Vj` inputs and reductions with `over="i"` work out of the box.

## Building ranges by hand

`kp.Ranges(ranges_i, slices_i, redranges_j, ranges_j=None, slices_j=None,
redranges_i=None)` takes the same six integer tensors as PyKeOps
(int32 or int64, on the device of the inputs):

| Tensor | Shape | Content |
| --- | --- | --- |
| `ranges_i` | `(Ci, 2)` | blocks of rows `[start, end)` of `i` |
| `slices_i` | `(Ci,)` | for each block, end position of its intervals in `redranges_j` (cumulative) |
| `redranges_j` | `(R, 2)` | intervals `[start, end)` of `j` to reduce over |
| `ranges_j`, `slices_j`, `redranges_i` | | the same in the other direction |

The second triplet is optional. Without it, `over="i"` reductions and
gradients with respect to `Vj` inputs raise an explicit error.

## Limitations

Ranges cannot be combined with [batch dimensions](batch.md).
