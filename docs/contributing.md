# Contributing

## Development install

```sh
git clone https://github.com/keckj/kernox
cd kernox
pip install "symox @ git+https://github.com/keckj/symox@53d9576e9f39db16e9ca8b3483d754385ebaf986"
pip install torch --index-url https://download.pytorch.org/whl/cu126   # or rocm10.0, xpu
pip install -e ".[cuda,test]"                                          # or [rocm,test], [xpu,test]
```

## Running the tests

```sh
pytest tests --ignore=tests/conformance                        # core tests, CPU oracle, no GPU needed
pytest -m "not slow" tests/conformance                         # conformance, on the GPU backends present
KP_CONFORMANCE_BACKENDS=hip pytest -m slow tests/conformance   # long profile, one backend
```

The core tests run everywhere: they use the CPU oracle. The examples of this
documentation are tests too (`tests/test_docs_examples.py`): every `python` block runs on the oracle, unless it is
preceded by a `<!-- skip-test -->` comment.

GPU tests are correctness tests and can run on a shared GPU, but not on a GPU
that is running a benchmark, since they would distort its timings.

## Building the documentation

```sh
pip install mkdocs-material "mkdocstrings[python]"
mkdocs build --strict
```

The home page is generated from `README.md`. The API reference is generated
from the docstrings.

## Where things live

| Path | Content |
| --- | --- |
| `kernox/expr.py`, `ops.py`, `complex.py` | formulas and reductions (the frontend) |
| `kernox/plan.py`, `builder.py` | the plan, the data structure passed to backends |
| `kernox/runtime.py` | `compile`, input validation, backend selection |
| `kernox/autograd.py` | gradients |
| `kernox/codegen/pair.py` | generation of the per-pair device function |
| `kernox/oracle.py` | the CPU reference |
| `kernox/backends/{cuda,hip,sycl}/` | the GPU backends |
| `kernox/tuning.py` | autotuning and wisdom |
| `tests/conformance/` | the conformance suite shared by the backends |

The technical pages that follow describe the rules every backend implements
([Core and backend contract](kernel-backend-contract.md)), the
[HIP backend](backend-hip.md) in detail, and the
[conformance suite](conformance.md).

General improvements to the symbolic engine and to code generation belong in
[symox](https://github.com/keckj/symox).
