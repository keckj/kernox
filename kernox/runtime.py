"""kernox runtime: ``compile``, input validation, backend registry.

All validation (names, shapes, dtypes, device, contiguity, K, gradients)
happens **before** any native launch. No fallback: a missing backend or an
unsupported operation raises an explicit error. The ``"oracle"`` backend
(blockwise Torch reference) is only used when explicitly requested;
``backend="auto"`` never selects the oracle.

Registry: a native backend is a ``kernox.backends.<name>`` module that
exposes ``get_backend() -> Backend`` (loaded lazily), or an object
registered with `register_backend`.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import math
import numbers
import threading
import time

from . import tuning as _tuning
from .backends.base import BatchLayout, RangesLayout, Sizes
from .builder import convert_float
from .codegen.pair import GENERATOR_VERSION
from .errors import BackendUnavailableError, UnsupportedError, ValidationError
from .plan import Plan, ReductionKind, output_shapes

NATIVE_BACKENDS = ("cuda", "hip", "sycl", "cpu")
# reductions with floating-point accumulation, affected by sum_scheme (amendment v0.7)
_SUMMING_KINDS = (ReductionKind.SUM, ReductionKind.LOGSUMEXP, ReductionKind.SUMSOFTMAXWEIGHT,
                  ReductionKind.LOGSUMEXPWEIGHT)
_K_KINDS = (ReductionKind.KMIN, ReductionKind.ARGKMIN, ReductionKind.KMIN_ARGKMIN)
_FACTORIES: dict = {}
_INSTANCES: dict = {}
_LOCK = threading.RLock()


def register_backend(name: str, factory, *, replace: bool = False) -> None:
    """Registers ``factory() -> Backend`` under ``name``."""
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"invalid backend name {name!r}")
    with _LOCK:
        if name in _FACTORIES and not replace:
            raise ValueError(f"backend {name!r} is already registered")
        _FACTORIES[name] = factory
        _INSTANCES.pop(name, None)


def _oracle_factory():
    from .oracle import OracleBackend

    return OracleBackend()


register_backend("oracle", _oracle_factory)


def get_backend(name: str):
    """(Cached) instance of backend ``name``; explicit error if missing."""
    with _LOCK:
        if name in _INSTANCES:
            return _INSTANCES[name]
        factory = _FACTORIES.get(name)
        if factory is None and name in NATIVE_BACKENDS:
            try:
                module = importlib.import_module(f"kernox.backends.{name}")
            except ImportError as exc:
                raise BackendUnavailableError(
                    f"backend {name!r} unavailable (module kernox.backends.{name}: {exc}); "
                    "no fallback is attempted"
                ) from exc
            factory = getattr(module, "get_backend", None)
            if factory is None:
                raise BackendUnavailableError(
                    f"kernox.backends.{name} does not expose get_backend(); no fallback is attempted"
                )
        if factory is None:
            raise BackendUnavailableError(
                f"unknown backend {name!r}; known: {sorted(set(_FACTORIES) | set(NATIVE_BACKENDS))}"
            )
        try:
            backend = factory()
        except Exception as exc:
            raise BackendUnavailableError(f"cannot initialize backend {name!r}: {exc}") from exc
        _INSTANCES[name] = backend
        return backend


def auto_backend_name(device) -> str:
    """Native backend for a Torch device: cuda (NVIDIA), hip (ROCm), sycl (XPU), cpu."""
    import torch

    if device.type == "cuda":
        return "hip" if getattr(torch.version, "hip", None) else "cuda"
    if device.type == "xpu":
        return "sycl"
    if device.type == "cpu":
        return "cpu"
    raise UnsupportedError(
        f"no native backend for device {device}; the Torch reference must be requested "
        "explicitly with backend='oracle'"
    )



def _device_fingerprint(device) -> dict:
    """Hardware and software identity of the Torch device (UUID if available)."""
    import torch

    fp = {"type": device.type, "torch": torch.__version__}
    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        fp.update(name=props.name, uuid=str(getattr(props, "uuid", "")),
                  runtime=torch.version.hip or torch.version.cuda)
    elif device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        props = torch.xpu.get_device_properties(device)
        fp.update(name=props.name, driver=str(getattr(props, "driver_version", "")))
    elif device.type == "cpu":
        from .backends.cpu import cpu_name

        fp.update(name=cpu_name())
    return fp


def torch_dtype(name: str):
    import torch

    return {"float16": torch.float16, "float32": torch.float32, "float64": torch.float64, "int64": torch.int64}[name]


_BROADCASTS = {}  # batch shapes of the inputs -> broadcast shape (bounded)


def _broadcast_batch(shapes: tuple) -> tuple:
    """``torch.broadcast_shapes`` in pure Python, cached (host cost per batched call)."""
    hit = _BROADCASTS.get(shapes)
    if hit is not None:
        return hit
    rank = max(map(len, shapes), default=0)
    out = [1] * rank
    for shape in shapes:
        lead = rank - len(shape)
        for k, size in enumerate(shape):
            cur = out[lead + k]
            if size != 1:
                if cur not in (1, size):
                    raise ValidationError(f"batch dimensions cannot be broadcast: {list(shapes)}")
                out[lead + k] = size
    res = tuple(out)
    if len(_BROADCASTS) >= 256:
        _BROADCASTS.clear()
    _BROADCASTS[shapes] = res
    return res


class Reduction:
    """Compiled reduction (callable). Call: ``run(x=..., y=..., sigma=0.7)``.

    Returns: a tensor (SUM, COUNT, MIN, MAX, ARG*, KMIN, ARGKMIN, LOGSUMEXP*,
    SUMSOFTMAXWEIGHT) or the tuple ``(values, indices)`` for KMIN_ARGKMIN,
    MIN_ARGMIN and MAX_ARGMAX.
    """

    def __init__(self, plan: Plan, backend: str, tuning: str = "estimate"):
        if backend != "auto" and not isinstance(backend, str):
            raise ValidationError(f"invalid backend {backend!r}")
        self.plan = plan
        self.backend = backend
        self.tuning = _tuning.check_level(tuning)
        self._compiled = {}       # (backend, device, variant) -> compiled object
        self._decisions = {}      # (backend, device, size class) -> (variant, decision)
        self._identities = {}     # (backend, device) -> wisdom identity
        self._warned = set()
        self._lock = threading.RLock()
        self.last_decision = None  # TuningDecision of the last launched call
        # Call-path caches (host cost); no validation is skipped: only values
        # derived from the plan, the device or the size are memoized.
        self._names = frozenset(a.name for a in plan.args)
        self._resolved = {}       # str(device) -> (name, backend, has variants)
        self._defaults = {}       # (name, device, M, N) -> default variant (bounded)
        self._params = {}         # (arg, values, device) -> parameter tensor (bounded)
        self._shapes = {}         # M -> Torch output shapes and dtypes (bounded)
        self._grads = {}          # frozenset(args) -> (GradientPlans, Reduction i, Reduction j)
        self._batched = {}        # batch rank -> Reduction of the batch_rank=r plan (amendment v0.6)
        self._layouts = {}        # (device, B, shapes, strides) -> BatchLayout (bounded)
        self._ranged = None       # Reduction of the ranges=True plan (amendment v0.6 §2)
        self._range_layouts = {}  # identity of the range tensors, tile_m -> RangesLayout (bounded)
        self._dt = None
        if backend != "auto":
            get_backend(backend)  # explicit error at compile() time if missing

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.plan.args)

    def _resolve(self, device):
        hit = self._resolved.get(str(device))
        if hit is not None:
            return hit[0], hit[1]
        name, backend = self._resolve_uncached(device)
        self._resolved[str(device)] = (name, backend, _tuning.backend_has_variants(backend))
        return name, backend

    def _resolve_uncached(self, device):
        name = auto_backend_name(device) if self.backend == "auto" else self.backend
        backend = get_backend(name)
        if not self.plan.fp_contract and not getattr(backend, "fp_contract_off", False):
            raise UnsupportedError(
                f"backend {name!r}: fp_contract=False (no FMA contraction) is not supported "
                "by this backend (contract v0.3)")
        if self.plan.flush_denormals and not getattr(backend, "flush_denormals", False):
            raise UnsupportedError(
                f"backend {name!r}: flush_denormals=True is not supported by this backend (amendment v0.11)")
        support = backend.supports(self.plan, device)
        if not support.ok:
            raise UnsupportedError(f"backend {name!r}: {support.reason}")
        scheme = self.plan.sum_scheme
        if (scheme != "direct" and self.plan.reduction.kind in _SUMMING_KINDS
                and scheme not in getattr(backend, "sum_schemes", ("direct",))):
            raise UnsupportedError(
                f"backend {name!r}: sum_scheme={scheme!r} is not supported (amendment v0.7, "
                "no fallback to 'direct')")
        if self.plan.ranges and not getattr(backend, "ranges", False):
            raise UnsupportedError(
                f"backend {name!r}: block-sparse ranges (amendment v0.6) are not supported")
        if self.plan.batch_rank and not getattr(backend, "batch", False):
            raise UnsupportedError(
                f"backend {name!r}: batch dimensions (amendment v0.6) are not supported")
        return name, backend

    def _compile(self, name, backend, device, variant):
        key = (name, str(device), variant)
        with self._lock:
            if key not in self._compiled:
                if variant is None:
                    self._compiled[key] = backend.compile(self.plan, device)
                else:
                    self._compiled[key] = backend.compile(self.plan, device, variant)
            return self._compiled[key]

    def compiled(self, device, m: int = 1, n: int = 1):
        """Compiled object (``estimate`` variant) for ``device`` and sizes M, N."""
        name, backend = self._resolve(device)
        if not _tuning.backend_has_variants(backend):
            return self._compile(name, backend, device, None)
        sizes = Sizes(m, n)
        return self._compile(name, backend, device, backend.default_variant(self.plan, device, sizes))

    def _identity(self, name, backend, device, default):
        key = (name, str(device))
        if key not in self._identities:
            ident = getattr(backend, "device_identity", None)
            if callable(ident):
                identity = dict(ident(device))
            else:
                info = self._compile(name, backend, device, default).info
                identity = {"device_name": info.device_name, "arch": info.arch, "compiler": info.compiler}
            identity.update({"backend": name, "generator": GENERATOR_VERSION, "device": _device_fingerprint(device)})
            self._identities[key] = json.loads(json.dumps(identity, sort_keys=True, default=str))
        return self._identities[key]

    def _select(self, device, tensors, m, n, new_outputs, nb: int = 1, density=None, launch_kwargs=None):
        """Variant and compiled object for this call, according to ``self.tuning``.

        Wisdom v3: ``nb`` (batch) and ``density()`` (range density, computed
        only outside ``estimate``: host synchronization) are part of the size
        class; ``launch_kwargs(compiled)`` completes the measured launches.
        """
        name, backend = self._resolve(device)
        has_variants = self._resolved[str(device)][2]
        level, plan = self.tuning, self.plan
        rho = density() if (density is not None and level != "estimate") else 1.0
        sclass = _tuning.size_class(m, n, nb, rho)
        if not has_variants:
            if level != "estimate" and name not in self._warned:
                self._warned.add(name)
                _tuning.log.warning("backend %r has no variants: tuning=%r falls back to 'estimate'", name, level)
            reason = "backend without variants" if level != "estimate" else ""
            decision = _tuning.TuningDecision(level, "fallback" if reason else "estimate", None, sclass, reason=reason)
            return self._compile(name, backend, device, None), decision
        sizes = Sizes(m, n, nb)
        dk = (name, str(device), m, n, nb)
        hit = self._defaults.get(dk)
        if hit is None:
            default = backend.default_variant(plan, device, sizes)
            hit = (default, _tuning.TuningDecision("estimate", "estimate", default.as_json(), sclass))
            if len(self._defaults) >= 256:
                self._defaults.clear()
            self._defaults[dk] = hit
        default = hit[0]
        if level == "estimate":
            return self._compile(name, backend, device, default), hit[1]
        dkey = (name, str(device), sclass)
        with self._lock:
            if dkey in self._decisions:
                variant, decision = self._decisions[dkey]
                return self._compile(name, backend, device, variant), decision
            identity = self._identity(name, backend, device, default)
            stored = _tuning.lookup(plan.key(), name, identity, sclass, level)
            if stored is not None:
                variant, decision = stored, _tuning.TuningDecision(level, "wisdom", stored.as_json(), sclass, persisted=True)
            elif level == "wisdom_only":
                variant = default
                decision = _tuning.TuningDecision(level, "estimate", default.as_json(), sclass, reason="no wisdom")
            else:
                cands, truncated = _tuning.candidates(backend, plan, device, sizes, level, default)
                variant, results, stable, seconds = _tuning.measure(
                    level, lambda v: self._compile(name, backend, device, v), cands, tensors, new_outputs, device,
                    launch_kwargs=launch_kwargs)
                decision = _tuning.TuningDecision(
                    level, "measured", variant.as_json(), sclass, stable=stable, candidates=len(cands),
                    truncated_from=truncated, measurements=results, tuning_seconds=seconds,
                    budget_seconds=_tuning.BUDGET_SECONDS[level],
                    reason="" if stable else "high spread: not persisted")
                if stable:
                    _tuning.store(plan.key(), name, identity, sclass, variant, {
                        "level": level, "median_ms": min(r["median_ms"] for r in results),
                        "candidates": len(results), "created": time.time()})
                    decision.persisted = True
                else:
                    _tuning.log.warning("tuning=%r: unstable measurement, decision not persisted", level)
            self._decisions[dkey] = (variant, decision)
        return self._compile(name, backend, device, variant), decision

    @property
    def tuning_decisions(self) -> list:
        """Measured or loaded decisions (outside ``estimate``), per device and size class."""
        return [d for _, d in self._decisions.values()]

    # ---- validation
    def validate(self, kwargs: dict):
        """Returns ``(tensors, device, M, N, B)`` or raises ``ValidationError``.

        ``B``: broadcast batch shape, ``None`` without batch.
        """
        import torch

        plan = self.plan
        if kwargs.keys() != self._names:
            names = self._names
            missing = sorted(names - set(kwargs))
            unexpected = sorted(set(kwargs) - names)
            raise ValidationError(f"missing arguments {missing}, unexpected {unexpected}")
        dt = self._dt
        if dt is None:
            dt = self._dt = torch_dtype(plan.dtype)
        Tensor = torch.Tensor

        device = None
        for a in plan.args:
            if a.role != "param":
                v = kwargs[a.name]
                if not isinstance(v, Tensor):
                    raise ValidationError(f"{a.name!r}: expected torch.Tensor (L, {a.dim}), got {type(v).__name__}")
                device = v.device
                if v.dim() != 2 or plan.batch_rank:
                    return self._validate_batched(kwargs, device, v.dim() - 2)
                break
        tensors, m, n = {}, None, None
        for a in plan.args:
            v = kwargs[a.name]
            role = a.role
            if role == "param":
                if not isinstance(v, Tensor):
                    v = self._param_from_python(a, v, dt, device)
            elif not isinstance(v, Tensor):
                raise ValidationError(f"{a.name!r}: expected torch.Tensor (L, {a.dim}), got {type(v).__name__}")
            if v.dtype != dt:
                raise ValidationError(
                    f"{a.name!r}: dtype {v.dtype} instead of {dt} (no implicit conversion)"
                )
            if v.device != device:
                raise ValidationError(f"{a.name!r}: device {v.device} differs from {device}")
            shape = v.shape
            if role == "param":
                if len(shape) == 0 and a.dim == 1:
                    v = v.reshape(1)
                    shape = v.shape
                if len(shape) != 1 or shape[0] != a.dim:
                    raise ValidationError(f"{a.name!r}: shape {tuple(shape)} instead of ({a.dim},)")
            else:
                if len(shape) != 2 or shape[1] != a.dim:
                    raise ValidationError(
                        f"{a.name!r}: shape {tuple(shape)} instead of (L, {a.dim}) (role {role})"
                    )
                length = shape[0]
                if role == "i":
                    if m is not None and m != length:
                        raise ValidationError(f"{a.name!r}: M={length} inconsistent with M={m}")
                    m = length
                else:
                    if n is not None and n != length:
                        raise ValidationError(f"{a.name!r}: N={length} inconsistent with N={n}")
                    n = length
            if not v.is_contiguous():
                raise ValidationError(f"{a.name!r}: non-contiguous tensor rejected (call .contiguous())")
            tensors[a.name] = v
        return tensors, device, m, n, None

    def _validate_batched(self, kwargs: dict, device, r: int):
        """Validation of inputs with batch dimensions (amendment v0.6)."""
        import torch

        plan = self.plan
        if r < 1 or (plan.batch_rank and r != plan.batch_rank):
            first = next(a for a in plan.args if a.role != "param")
            want = f"(*B, L, {first.dim})" if plan.batch_rank else f"(L, {first.dim}) or (*B, L, {first.dim})"
            raise ValidationError(f"{first.name!r}: shape {tuple(kwargs[first.name].shape)} instead of {want}")
        dt = self._dt
        if dt is None:
            dt = self._dt = torch_dtype(plan.dtype)
        Tensor = torch.Tensor
        tensors, m, n, batch_shapes = {}, None, None, []
        for a in plan.args:
            v = kwargs[a.name]
            role = a.role
            if role == "param":
                if not isinstance(v, Tensor):
                    v = self._param_from_python(a, v, dt, device)
            elif not isinstance(v, Tensor):
                raise ValidationError(f"{a.name!r}: expected torch.Tensor (*B, L, {a.dim}), got {type(v).__name__}")
            if v.dtype != dt:
                raise ValidationError(f"{a.name!r}: dtype {v.dtype} instead of {dt} (no implicit conversion)")
            if v.device != device:
                raise ValidationError(f"{a.name!r}: device {v.device} differs from {device}")
            shape, st = tuple(v.shape), v.stride()
            if role == "param":
                if len(shape) == 0 and a.dim == 1:
                    v = v.reshape(1)
                    shape, st = (1,), (1,)
                if len(shape) not in (1, r + 1) or shape[-1] != a.dim:
                    raise ValidationError(
                        f"{a.name!r}: shape {shape} instead of ({a.dim},) or (*B, {a.dim}) "
                        f"with {r} batch dimension(s)")
                if a.dim > 1 and st[-1] != 1:
                    raise ValidationError(f"{a.name!r}: non-contiguous last dimension rejected")
                if len(shape) > 1:
                    batch_shapes.append(shape[:-1])
            else:
                if len(shape) != r + 2 or shape[-1] != a.dim:
                    raise ValidationError(
                        f"{a.name!r}: shape {shape} instead of (*B, L, {a.dim}) with {r} batch "
                        f"dimension(s) (rank shared by all i and j inputs, role {role})")
                length = shape[-2]
                if role == "i":
                    if m is not None and m != length:
                        raise ValidationError(f"{a.name!r}: M={length} inconsistent with M={m}")
                    m = length
                else:
                    if n is not None and n != length:
                        raise ValidationError(f"{a.name!r}: N={length} inconsistent with N={n}")
                    n = length
                if (a.dim > 1 and st[-1] != 1) or (length > 1 and st[-2] != a.dim):
                    raise ValidationError(
                        f"{a.name!r}: non-contiguous last two dimensions rejected "
                        "(batch dimensions may have arbitrary strides)")
                batch_shapes.append(shape[:-2])
            tensors[a.name] = v
        return tensors, device, m, n, _broadcast_batch(tuple(batch_shapes))

    def _param_from_python(self, a, v, dt, device):
        import torch

        values = tuple(v) if isinstance(v, (list, tuple)) else (v,)
        if len(values) != a.dim:
            raise ValidationError(f"{a.name!r}: expected {a.dim} values, got {len(values)}")
        for x in values:
            if isinstance(x, bool) or not isinstance(x, numbers.Real):
                raise ValidationError(f"{a.name!r}: expected a real value, got {x!r}")
        # Tensor cached by exact values (type included): avoids one host -> device
        # copy per call. Inputs are read-only for the backends (contract), so
        # the shared tensor is never modified.
        key = (a.name, tuple((type(x), x) for x in values), str(device))
        hit = self._params.get(key)
        if hit is not None:
            return hit
        conv = [convert_float(float(x), str(dt).removeprefix("torch."), f"parameter {a.name}") for x in values]
        t = torch.tensor(conv, dtype=dt, device=device)
        if len(self._params) >= 64:
            self._params.clear()
        self._params[key] = t
        return t

    # ---- execution
    def __call__(self, **kwargs):
        meta = self.plan.meta
        if meta.get("complex_args") or meta.get("complex_output"):
            return _complex_call(self, self._call, kwargs)
        return self._call(**kwargs)

    def _call(self, **kwargs):
        import torch

        tensors, device, m, n, batch = self.validate(kwargs)
        if torch.is_grad_enabled() and any(t.requires_grad for t in tensors.values()):
            from . import autograd

            fn = autograd.function_for(self.plan.reduction.kind)
            return fn.apply(self, tuple(tensors), *tensors.values())
        if batch is None:
            return self._run(tensors, device, m, n)
        return self._run_batched(tensors, device, m, n, batch)

    # ---- autograd (see kernox.autograd)
    def _forward(self, tensors: dict):
        """Forward pass on already validated (detached) tensors."""
        import torch

        plan = self.plan
        device = m = n = None
        batched = bool(plan.batch_rank)
        shapes = []
        for a in plan.args:
            t = tensors[a.name]
            if a.role == "param":
                if t.dim() > 1:
                    shapes.append(t.shape[:-1])
                continue
            batched = batched or t.dim() != 2
            shapes.append(t.shape[:-2])
            if a.role == "i" and m is None:
                m, device = t.shape[-2], t.device
            elif a.role == "j" and n is None:
                n = t.shape[-2]
        if not batched:
            return self._run(tensors, device, m, n)
        return self._run_batched(tensors, device, m, n, tuple(torch.broadcast_shapes(*shapes)))

    def _gradient(self, wrt: frozenset):
        hit = self._grads.get(wrt)
        if hit is None:
            from .autograd import gradient_plans

            gp = gradient_plans(self.plan, wrt)
            hit = (gp,
                   Reduction(gp.i_plan, self.backend, self.tuning) if gp.i_plan is not None else None,
                   Reduction(gp.j_plan, self.backend, self.tuning) if gp.j_plan is not None else None)
            self._grads[wrt] = hit
        return hit

    def _backward(self, tensors: dict, grad_out, wrt, ranges=None):
        """Gradients ``{name: tensor}`` of the ``wrt`` inputs (fused reductions).

        ``ranges``: ``(forward triplet, transposed triplet)`` in the plan
        orientation (amendment v0.6 §2); the j side of the gradients uses the transposed one.
        """
        if not wrt:
            return {}
        gp, i_red, j_red = self._gradient(frozenset(wrt))
        dt = torch_dtype(self.plan.dtype)
        inputs = dict(tensors)
        inputs[gp.grad_name] = grad_out.to(dt).contiguous()
        grads = {}
        for side, red, layout in (("i", i_red, gp.i_layout), ("j", j_red, gp.j_layout)):
            if red is None:
                continue
            if ranges is None:
                out = red(**inputs)
            else:
                fwd, bwd = ranges if side == "i" else ranges[::-1]
                if fwd is None:
                    raise UnsupportedError(
                        "gradient with respect to a reduced argument with ranges: the transposed triplet "
                        "(ranges_j, slices_j, redranges_i in user orientation) is required")
                out = RangedReduction(red, fwd, bwd)(**inputs)
            batched = out.dim() > 2
            off = 0
            for name, role, dim in layout:
                g = out[..., off:off + dim]
                off += dim
                if role == "param":
                    g = g.sum(dim=-2)
                g = g.to(tensors[name].dtype)
                shape = tensors[name].shape
                # batch: sum over the broadcast dimensions of the input (amendment v0.6)
                grads[name] = g.sum_to_size(shape) if batched else g.reshape(shape)
        return grads

    def _backward_surrogate(self, tensors: dict, out, grad_out, wrt, ranges=None):
        """Gradients of LOGSUMEXP/LOGSUMEXPWEIGHT/SUMSOFTMAXWEIGHT (surrogate SUM plan)."""
        if not wrt:
            return {}
        hit = self._grads.get("surrogate")
        if hit is None:
            from .autograd import surrogate

            sur = surrogate(self.plan)
            hit = (sur, Reduction(sur.plan, self.backend, self.tuning),
                   Reduction(sur.lse_plan, self.backend, self.tuning) if sur.lse_plan is not None else None)
            self._grads["surrogate"] = hit
        sur, sred, lred = hit
        dt = torch_dtype(self.plan.dtype)
        inputs = {k: v.detach() for k, v in tensors.items()}
        for name, _dim, src in sur.fixed:
            if src == "out":
                v = out
            elif ranges is None:
                v = lred(**inputs)
            else:
                v = RangedReduction(lred, ranges[0], ranges[1])(**inputs)
            inputs[name] = v.detach().to(dt).contiguous()
        return sred._backward(inputs, grad_out, wrt, ranges=ranges)

    def _backward_select(self, tensors: dict, index, grad_value, wrt, ranges=None):
        """Gradients of MIN/MAX/KMIN at the selected term (indices from the forward pass, or recomputed)."""
        if not wrt:
            return {}
        from .autograd import SelectGradient, index_plan

        inputs = {k: v.detach() for k, v in tensors.items()}
        if index is None:
            ired = self._grads.get("index")
            if ired is None:
                ired = self._grads["index"] = Reduction(index_plan(self.plan), self.backend, self.tuning)
            index = ired(**inputs) if ranges is None else RangedReduction(ired, ranges[0], ranges[1])(**inputs)
        key = ("select", frozenset(wrt))
        sel = self._grads.get(key)
        if sel is None:
            sel = self._grads[key] = SelectGradient(self.plan, wrt)
        return sel(inputs, index, grad_value)

    def _run(self, tensors, device, m, n):
        import torch

        plan = self.plan
        compiled = None
        shapes = self._shapes.get(m)
        if shapes is None:
            shapes = tuple((key, shape, torch_dtype(dtype)) for key, (shape, dtype) in output_shapes(plan, m).items())
            if len(self._shapes) >= 64:
                self._shapes.clear()
            self._shapes[m] = shapes
        empty_ = torch.empty
        outputs = {key: empty_(shape, dtype=dt, device=device) for key, shape, dt in shapes}
        k = plan.reduction.k
        empty = m == 0 or (k == 0 and plan.reduction.kind in (
            ReductionKind.KMIN, ReductionKind.ARGKMIN, ReductionKind.KMIN_ARGKMIN))
        if not empty:
            def new_outputs():
                return {key: empty_(shape, dtype=dt, device=device) for key, shape, dt in shapes}

            compiled, decision = self._select(device, tensors, m, n, new_outputs)
            compiled.launch(tensors, outputs)  # always the selected variant
            self.last_decision = decision
        else:
            # the backend must exist and accept the plan even without a launch
            self._resolve(device)
        if len(outputs) == 2:  # KMIN_ARGKMIN, MIN_ARGMIN, MAX_ARGMAX
            return outputs["value"], outputs["index"]
        return next(iter(outputs.values()))

    # ---- batch (amendment v0.6)
    def _run_batched(self, tensors, device, m, n, B):
        r = len(B)
        if self.plan.batch_rank == r:
            return self._launch_batched(tensors, device, m, n, B)
        child = self._batched.get(r)
        if child is None:
            child = Reduction(dataclasses.replace(self.plan, batch_rank=r), self.backend, self.tuning)
            self._batched[r] = child
        return child._launch_batched(tensors, device, m, n, B)

    def _layout(self, tensors, device, B, nb) -> BatchLayout:
        """Per-argument offset tables (host and device), cached by shapes and strides."""
        import torch

        key = (str(device), B, tuple((name, tuple(t.shape), t.stride()) for name, t in tensors.items()))
        hit = self._layouts.get(key)
        if hit is not None:
            return hit
        host, dev = {}, {}
        for a in self.plan.args:
            t = tensors[a.name]
            tail = 1 if a.role == "param" else 2
            bshape, bstride = tuple(t.shape[:-tail]), t.stride()[:-tail]
            off = torch.zeros(B, dtype=torch.int64)
            lead = len(B) - len(bshape)  # param without batch: empty bshape
            for k, (size, stride) in enumerate(zip(bshape, bstride)):
                if size != 1 and stride:
                    view = [1] * len(B)
                    view[lead + k] = size
                    off = off + torch.arange(size, dtype=torch.int64).view(view) * stride
            off = off.reshape(nb).contiguous()
            host[a.name] = off
            dev[a.name] = off if torch.device(device).type == "cpu" else off.to(device)
        layout = BatchLayout(B, nb, dev, host)
        if len(self._layouts) >= 64:
            self._layouts.clear()
        self._layouts[key] = layout
        return layout

    def _launch_batched(self, tensors, device, m, n, B):
        import torch

        plan = self.plan
        nb = math.prod(B)
        shapes = self._shapes.get((m, nb))
        if shapes is None:
            shapes = tuple((key, shape, torch_dtype(dt)) for key, (shape, dt) in output_shapes(plan, m, (nb,)).items())
            if len(self._shapes) >= 64:
                self._shapes.clear()
            self._shapes[(m, nb)] = shapes
        empty_ = torch.empty
        outputs = {key: empty_(shape, dtype=dt, device=device) for key, shape, dt in shapes}
        empty = nb == 0 or m == 0 or (plan.reduction.k == 0 and plan.reduction.kind in _K_KINDS)
        if not empty:
            layout = self._layout(tensors, device, B, nb)

            def new_outputs():
                return {key: empty_(shape, dtype=dt, device=device) for key, shape, dt in shapes}

            compiled, decision = self._select(device, tensors, m, n, new_outputs, nb,
                                              launch_kwargs=lambda c: {"batch": layout})
            compiled.launch(tensors, outputs, batch=layout)
            self.last_decision = decision
        else:
            self._resolve(device)  # the backend must exist and accept the batched plan
        outs = [o.view(*B, *o.shape[1:]) for o in outputs.values()]
        if len(outs) == 2:  # (values, indices)
            return outs[0], outs[1]
        return outs[0]

    # ---- block-sparse ranges (amendment v0.6 §2)
    # ---- linear operator and solve (PyKeOps parity, see kernox.linalg)
    def linear_operator(self, var: str, /, **inputs):
        """`LinearOperator` ``v ↦ self(**inputs, var=v)`` (SUM linear in ``var``)."""
        from .linalg import LinearOperator

        return LinearOperator(self, var, inputs)

    def solve(self, var: str, rhs, /, *, alpha: float = 1e-10, eps: float = 1e-6, maxiter=None, x0=None,
              inputs: dict | None = None, **kwargs):
        """Solves ``(alpha·I + K) var = rhs`` by conjugate gradient, with ``K = self`` viewed as linear in ``var``.

        The other inputs are passed by keyword (or ``inputs={...}`` if a
        name conflicts with ``alpha``, ``eps``, ``maxiter``, ``x0``).
        See `kernox.linalg.LinearOperator.solve`.
        """
        merged = dict(inputs or {})
        clash = merged.keys() & kwargs.keys()
        if clash:
            raise ValidationError(f"solve: inputs given twice {sorted(clash)}")
        merged.update(kwargs)
        return self.linear_operator(var, **merged).solve(rhs, alpha=alpha, eps=eps, maxiter=maxiter, x0=x0)

    def with_ranges(self, ranges, *, validate: bool = False) -> "RangedReduction":
        """Call restricted to the blocks of ``ranges`` (`kernox.Ranges`).

        ``run.with_ranges(r)(**inputs)``. ``validate=True`` checks the values
        (bounds, overlaps; one host synchronization per call).
        """
        from .ranges import Ranges

        if not isinstance(ranges, Ranges):
            raise ValidationError(f"with_ranges expects a kernox.Ranges, got {type(ranges).__name__}")
        fwd, bwd = ranges.oriented(bool(self.plan.meta.get("swapped_ij")))
        return RangedReduction(self, fwd, bwd, validate=validate)

    def _run_ranged(self, tensors, device, m, n, triplet):
        child = self._ranged
        if child is None:
            child = self._ranged = Reduction(dataclasses.replace(self.plan, ranges=True), self.backend, self.tuning)
        return child._launch_ranged(tensors, device, m, n, triplet)

    def _ranges_layout(self, triplet, device, tile_m, m=None, n=None) -> RangesLayout:
        """Ranges layout (cached by tensor identity); with ``m`` and ``n``, the
        bounds check is **always on**: a tile or an interval outside
        ``[0, m] × [0, n]`` (or an interval index outside ``redranges``) would make
        the kernel read or write outside the tensors, or loop over billions of j
        (loss of the device). The summary is computed once on the host copies of the
        layout, O(1) cost per call; ``validate=True`` adds the overlap checks."""
        layout = self._ranges_layout_cached(triplet, device, tile_m)
        if m is not None:
            s = layout.summary
            kok = s.nw == 0 or (s.k0_min >= 0 and s.dk_min >= 0 and s.k1_max <= s.nr)
            if not kok or s.live_i0_min < 0 or s.live_i1_max > m or s.live_c_min < 0 or s.live_d_max > n:
                raise ValidationError(
                    f"ranges out of bounds for M={m}, N={n}: rows [{s.live_i0_min}, {s.live_i1_max}), "
                    f"j [{s.live_c_min}, {s.live_d_max})"
                    + ("" if kok else ", slices outside redranges or decreasing"))
        return layout

    def _ranges_layout_cached(self, triplet, device, tile_m) -> RangesLayout:
        from .ranges import work_list

        key = (str(device), tile_m) + tuple(
            (t.data_ptr(), t._version, tuple(t.shape), str(t.dtype)) for t in (triplet.rows, triplet.slices, triplet.red))
        hit = self._range_layouts.get(key)
        if hit is not None and hit[0] is triplet.rows and hit[1] is triplet.red:
            return hit[2]
        import torch

        work = work_list(triplet, tile_m)
        red = triplet.red.to(torch.int64).contiguous()
        layout = RangesLayout(work, red, tile_m)  # host copies on demand (reference backends)
        if len(self._range_layouts) >= 64:
            self._range_layouts.clear()
        self._range_layouts[key] = (triplet.rows, triplet.red, layout)
        layout.summary  # noqa: B018 - computed once, off the hot path
        return layout

    def _launch_ranged(self, tensors, device, m, n, triplet):
        import torch

        plan = self.plan
        outputs = {key: torch.empty(shape, dtype=torch_dtype(dt), device=device)
                   for key, (shape, dt) in output_shapes(plan, m).items()}
        empty = m == 0 or (plan.reduction.k == 0 and plan.reduction.kind in _K_KINDS)
        if not empty:
            shapes = [(shape, torch_dtype(dt)) for shape, dt in output_shapes(plan, m).values()]
            keys = list(outputs)

            def new_outputs():
                return {k: torch.empty(s, dtype=d, device=device) for k, (s, d) in zip(keys, shapes)}

            compiled, decision = self._select(
                device, tensors, m, n, new_outputs, density=lambda: _ranges_density(triplet, m, n),
                launch_kwargs=lambda c: {"ranges": self._ranges_layout(triplet, device, int(c.ranges_tile_m), m, n)})
            tile_m = int(getattr(compiled, "ranges_tile_m"))
            compiled.launch(tensors, outputs, ranges=self._ranges_layout(triplet, device, tile_m, m, n))
            self.last_decision = decision
        else:
            self._resolve(device)
        if len(outputs) == 2:  # KMIN_ARGKMIN, MIN_ARGMIN, MAX_ARGMAX
            return outputs["value"], outputs["index"]
        return next(iter(outputs.values()))

    def __repr__(self):
        return f"kernox.Reduction({self.plan.reduction.kind.value}, backend={self.backend!r}, dtype={self.plan.dtype})"


def _ranges_density(triplet, m: int, n: int) -> float:
    """Fraction ρ of the (i, j) pairs reduced by the triplet (wisdom v3; host synchronization)."""
    import torch

    rows, slices, red = (t.detach().to("cpu", torch.int64) for t in (triplet.rows, triplet.slices, triplet.red))
    if m * n == 0 or rows.shape[0] == 0:
        return 0.0
    lens = (red[:, 1] - red[:, 0]).clamp(min=0)
    cum = torch.cat([torch.zeros(1, dtype=torch.int64), lens.cumsum(0)])
    starts = torch.cat([torch.zeros(1, dtype=torch.int64), slices[:-1]])
    per_block = cum[slices] - cum[starts]
    pairs = int(((rows[:, 1] - rows[:, 0]).clamp(min=0) * per_block).sum())
    return min(1.0, pairs / (m * n))


def _complex_call(red, call, kwargs):
    """Complex inputs viewed as interleaved reals; a complex SUM output is returned as complex."""
    from .complex import complex_input, complex_output

    plan = red.plan
    dims = {a.name: a.dim for a in plan.args}
    kw = dict(kwargs)
    for name in plan.meta.get("complex_args", ()):
        if name in kw and name in dims:
            kw[name] = complex_input(name, kw[name], dims[name])
    out = call(**kw)
    return complex_output(out) if plan.meta.get("complex_output") else out


class RangedReduction:
    """Reduction restricted by block-sparse ranges (``run.with_ranges(r)``).

    Triplets in the plan orientation: ``fwd`` for the forward pass,
    ``bwd`` (transposed, possibly None) for the j-side gradients.
    """

    def __init__(self, red: Reduction, fwd, bwd=None, *, validate: bool = False):
        self.red, self.plan = red, red.plan
        self._fwd, self._bwd, self._validate = fwd, bwd, validate

    def _check(self, device, m, n):
        for tr, rows, cols, side in ((self._fwd, m, n, "forward"), (self._bwd, n, m, "transposed")):
            if tr is None:
                continue
            if tr.rows.device != device:
                raise ValidationError(f"ranges ({side}): device {tr.rows.device} differs from {device}")
            if self._validate:
                from .ranges import validate_triplet

                validate_triplet(tr, rows, cols, side)

    def __call__(self, **kwargs):
        meta = self.plan.meta
        if meta.get("complex_args") or meta.get("complex_output"):
            return _complex_call(self, self._call, kwargs)
        return self._call(**kwargs)

    def _call(self, **kwargs):
        import torch

        tensors, device, m, n, batch = self.red.validate(kwargs)
        if batch is not None:
            raise UnsupportedError("ranges combined with batch dimensions: rejected (amendment v0.6)")
        self._check(device, m, n)
        if torch.is_grad_enabled() and any(t.requires_grad for t in tensors.values()):
            from . import autograd

            fn = autograd.function_for(self.plan.reduction.kind)
            return fn.apply(self, tuple(tensors), *tensors.values())
        return self._run(tensors, device, m, n)

    def _run(self, tensors, device, m, n):
        return self.red._run_ranged(tensors, device, m, n, self._fwd)

    def _forward(self, tensors: dict):
        plan = self.plan
        m = n = device = None
        for a in plan.args:
            t = tensors[a.name]
            if a.role == "i" and m is None:
                m, device = t.shape[0], t.device
            elif a.role == "j" and n is None:
                n = t.shape[0]
        return self._run(tensors, device, m, n)

    def _backward(self, tensors: dict, grad_out, wrt):
        return self.red._backward(tensors, grad_out, wrt, ranges=(self._fwd, self._bwd))

    def _backward_surrogate(self, tensors: dict, out, grad_out, wrt):
        return self.red._backward_surrogate(tensors, out, grad_out, wrt, ranges=(self._fwd, self._bwd))

    def _backward_select(self, tensors: dict, index, grad_value, wrt):
        return self.red._backward_select(tensors, index, grad_value, wrt, ranges=(self._fwd, self._bwd))

    linear_operator = Reduction.linear_operator
    solve = Reduction.solve

    def __repr__(self):
        return f"kernox.RangedReduction({self.red!r})"


def compile(expr, *, dtype, backend: str = "auto", acc_dtype=None, tuning: str = "estimate",  # noqa: A001
            fp_contract: bool | None = None, sum_scheme: str | None = None,
            fast_math: bool | None = None, flush_denormals: bool | None = None) -> Reduction:
    """Compiles a frontend reduction (or a `Plan`) into a callable.

    ``dtype`` is mandatory (``"float16"``, ``"float32"`` or ``"float64"``): no
    implicit precision. ``backend``: ``"auto"`` (device of the tensors:
    cuda/hip/sycl/cpu), a native name, or ``"oracle"`` (explicit reference).
    ``tuning``: ``"estimate"`` (default, no measurement), ``"wisdom_only"``,
    ``"measure"`` or ``"patient"`` (see `kernox.tuning`).
    ``fp_contract``: ``True`` by default, FMA contraction
    allowed within an expression; ``False`` forbids it on all
    backends (bit-for-bit reproducibility; a backend that cannot
    guarantee it raises `UnsupportedError`). Part of the plan key.
    ``sum_scheme``: ``"direct"`` (default), ``"block"`` or
    ``"kahan"`` (Neumaier) for SUM, LOGSUMEXP and SUMSOFTMAXWEIGHT; ignored by
    the other reductions. A backend that does not implement it raises
    `UnsupportedError`, never a fallback to ``"direct"``.
    ``fast_math``: ``False`` by default (IEEE); ``True`` allows
    approximate exp/log (CUDA FP32: ``ex2.approx``/``lg2.approx``, like PyKeOps
    ``use_fast_math``). Part of the plan key; a backend that does not
    implement it raises `UnsupportedError`. The oracle stays IEEE.
    ``flush_denormals``: ``False`` by default (subnormal numbers kept);
    ``True`` allows the backend to flush subnormal inputs and results of
    floating-point operations to zero (cpu: FTZ/DAZ; cuda: ``--ftz=true``,
    float32 only). Part of the plan key; a backend that does not implement it
    raises `UnsupportedError`. The oracle keeps subnormal numbers.
    """
    from .expr import ReductionExpr

    if fast_math is not None and not isinstance(fast_math, bool):
        raise ValidationError(f"fast_math must be a boolean, got {fast_math!r}")
    if fp_contract is not None and not isinstance(fp_contract, bool):
        raise ValidationError(f"fp_contract must be a boolean, got {fp_contract!r}")
    if flush_denormals is not None and not isinstance(flush_denormals, bool):
        raise ValidationError(f"flush_denormals must be a boolean, got {flush_denormals!r}")
    if isinstance(expr, ReductionExpr):
        plan = expr.to_plan(dtype, acc_dtype=acc_dtype, fp_contract=True if fp_contract is None else fp_contract,
                            sum_scheme=sum_scheme, fast_math=bool(fast_math),
                            flush_denormals=bool(flush_denormals))
    elif isinstance(expr, Plan):
        if dtype != expr.dtype or (acc_dtype is not None and acc_dtype != expr.acc_dtype):
            raise ValidationError("dtype/acc_dtype inconsistent with the given plan")
        if fp_contract is not None and fp_contract != expr.fp_contract:
            raise ValidationError("fp_contract inconsistent with the given plan")
        if fast_math is not None and fast_math != expr.fast_math:
            raise ValidationError("fast_math inconsistent with the given plan")
        if flush_denormals is not None and flush_denormals != expr.flush_denormals:
            raise ValidationError("flush_denormals inconsistent with the given plan")
        if sum_scheme is not None and sum_scheme != expr.sum_scheme:
            raise ValidationError("sum_scheme inconsistent with the given plan")
        plan = expr
    else:
        raise TypeError(
            "compile expects a reduction (e.g. formula.sum_over('j')) or a Plan, "
            f"got {type(expr).__name__}"
        )
    return Reduction(plan, backend, tuning)


__all__ = ["compile", "Reduction", "RangedReduction", "register_backend", "get_backend", "auto_backend_name", "NATIVE_BACKENDS"]
