# Autotuning and wisdom

For each reduction, a backend can generate several kernel variants (block
size, how the work over `j` is split, how the K-nearest lists are shared
between threads...). By default kernox picks one with a fixed heuristic. You
can instead let it measure a few variants on your device and remember the
fastest, in the spirit of FFTW's planning levels and "wisdom".

```python
import torch
import kernox as kp

device = "cuda"

x, y = kp.Vi("x", 3), kp.Vj("y", 3)
knn = kp.compile(x.sqdist(y).argkmin(16, over="j"), dtype="float32", tuning="measure")

X = torch.randn(2_000, 3, device=device)
Y = torch.randn(5_000, 3, device=device)
idx = knn(x=X, y=Y)            # first call of this size: measures, then runs
idx = knn(x=X, y=Y)            # later calls: no measurement
print(knn.last_decision)
```

## Tuning levels

| `tuning` | Behaviour |
| --- | --- |
| `"estimate"` (default) | heuristic choice, no measurement; results are reproducible on a given device |
| `"wisdom_only"` | use a stored decision if there is one, otherwise `"estimate"`; never measures |
| `"measure"` | use a stored decision, or measure up to 4 variants (about 2 s at most) |
| `"patient"` | same, with up to 32 variants (about 30 s at most) |

Changing variant never changes the meaning of the result: same dtypes, same
tie and NaN rules. Only float sums can differ slightly, because the terms are
added in a different order.

## When measurements happen

- Only on the first call for a given **size class**. Sizes are grouped by
  factors of about $\sqrt{2}$ for $M$, $N$, the number of batch elements and the density
  of block-sparse ranges. Other calls in the same class reuse the decision
  without measuring or recompiling.
- The measuring call is **blocking**: it synchronizes with the GPU. It
  measures on the real inputs of the call (read-only), and the call returns the
  result computed with the chosen variant.
- If the timings are too noisy (for example because the GPU is busy), the
  decision is kept for this compiled object only and not stored.

Measure on an idle GPU, so that the stored decisions reflect the real speed.

## Inspecting decisions

`run.last_decision` describes the decision of the last call: the level, its
source (`"estimate"`, `"wisdom"`, `"measured"` or `"fallback"`), the chosen
variant, the size class, whether it was stable and stored, and the measured
times. `run.tuning_decisions` lists all decisions of a compiled object.

A backend without variants (such as the CPU oracle) falls back to
`"estimate"` and logs a message once.

On the `cpu` backend the variants change the `j` tile and the number of rows
per task. The default already adapts the tile to the L1 cache, and measured
gains from tuning are small (a few percent); see
[CPU backend](../backend-cpu.md#variants).

## Wisdom files

Stored decisions ("wisdom") are JSON files, one per device, in
`$KERNOX_CACHE_DIR/wisdom` (by default `~/.cache/kernox/wisdom`). A decision is
only reused on the same device (identified by its UUID when the driver
reports one), with the same driver, compiler and kernox code generator, and
only by a level at most as demanding as the one that produced it: `"patient"`
does not trust a decision made by `"measure"`.

To save, restore or reset the wisdom:

```python
data = kp.export_wisdom()                      # all valid entries, as a dict
kp.export_wisdom("wisdom.json")                # the same, written to a file
n_imported = kp.import_wisdom("wisdom.json")   # also accepts the dict
n_deleted = kp.forget_wisdom()                 # deletes the wisdom files
```

`import_wisdom` skips invalid entries and returns the number of entries
imported; a file of another wisdom format raises `ValueError`.
