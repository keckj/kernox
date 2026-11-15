"""kernox runtime: ``compile``, input validation, backend registry.

All validation (names, shapes, dtypes, device, contiguity, K, gradients)
happens **before** any native launch. No fallback: a missing backend or an
unsupported operation raises an explicit error. The ``"oracle"`` backend
(blockwise Torch reference) is only used when explicitly requested;
``backend="auto"`` never selects the oracle.

Registry: a native backend is a ``kernox.backends.<name>`` module that
exposes ``get_backend() -> Backend`` (loaded lazily), or an object
registered with :func:`register_backend`.
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

NATIVE_BACKENDS = ("cuda", "hip", "sycl")
# réductions à accumulation flottante, concernées par sum_scheme (avenant v0.7)
_SUMMING_KINDS = (ReductionKind.SUM, ReductionKind.LOGSUMEXP, ReductionKind.SUMSOFTMAXWEIGHT,
                  ReductionKind.LOGSUMEXPWEIGHT)
_K_KINDS = (ReductionKind.KMIN, ReductionKind.ARGKMIN, ReductionKind.KMIN_ARGKMIN)
_FACTORIES: dict = {}
_INSTANCES: dict = {}
_LOCK = threading.RLock()


def register_backend(name: str, factory, *, replace: bool = False) -> None:
    """Registers ``factory() -> Backend`` under ``name``."""
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"nom de backend invalide {name!r}")
    with _LOCK:
        if name in _FACTORIES and not replace:
            raise ValueError(f"backend {name!r} déjà enregistré")
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
                    f"backend {name!r} indisponible (module kernox.backends.{name} : {exc}) ; "
                    "aucun repli n'est effectué"
                ) from exc
            factory = getattr(module, "get_backend", None)
            if factory is None:
                raise BackendUnavailableError(
                    f"kernox.backends.{name} n'expose pas get_backend() ; aucun repli n'est effectué"
                )
        if factory is None:
            raise BackendUnavailableError(
                f"backend inconnu {name!r} ; connus : {sorted(set(_FACTORIES) | set(NATIVE_BACKENDS))}"
            )
        try:
            backend = factory()
        except Exception as exc:
            raise BackendUnavailableError(f"initialisation du backend {name!r} impossible : {exc}") from exc
        _INSTANCES[name] = backend
        return backend


def auto_backend_name(device) -> str:
    """Native backend for a Torch device: cuda (NVIDIA), hip (ROCm), sycl (XPU)."""
    import torch

    if device.type == "cuda":
        return "hip" if getattr(torch.version, "hip", None) else "cuda"
    if device.type == "xpu":
        return "sycl"
    raise UnsupportedError(
        f"aucun backend GPU pour le device {device} ; la référence CPU doit être demandée "
        "explicitement avec backend='oracle'"
    )



def _device_fingerprint(device) -> dict:
    """Identité matérielle et logicielle du device Torch (UUID si disponible)."""
    import torch

    fp = {"type": device.type, "torch": torch.__version__}
    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        fp.update(name=props.name, uuid=str(getattr(props, "uuid", "")),
                  runtime=torch.version.hip or torch.version.cuda)
    elif device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        props = torch.xpu.get_device_properties(device)
        fp.update(name=props.name, driver=str(getattr(props, "driver_version", "")))
    return fp


def torch_dtype(name: str):
    import torch

    return {"float16": torch.float16, "float32": torch.float32, "float64": torch.float64, "int64": torch.int64}[name]


_BROADCASTS = {}  # formes de batch des entrées -> forme diffusée (bornée)


def _broadcast_batch(shapes: tuple) -> tuple:
    """``torch.broadcast_shapes`` en Python pur, mis en cache (coût hôte par appel batché)."""
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
                    raise ValidationError(f"dimensions de batch non diffusables : {list(shapes)}")
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
            raise ValidationError(f"backend invalide {backend!r}")
        self.plan = plan
        self.backend = backend
        self.tuning = _tuning.check_level(tuning)
        self._compiled = {}       # (backend, device, variant) -> objet compilé
        self._decisions = {}      # (backend, device, classe de taille) -> (variante, décision)
        self._identities = {}     # (backend, device) -> identité de wisdom
        self._warned = set()
        self._lock = threading.RLock()
        self.last_decision = None  # TuningDecision du dernier appel lancé
        # Caches du chemin d'appel (coût hôte) ; aucune validation n'est sautée :
        # seules des valeurs dérivées du plan, du device ou de la taille sont mémorisées.
        self._names = frozenset(a.name for a in plan.args)
        self._resolved = {}       # str(device) -> (nom, backend, a des variantes)
        self._defaults = {}       # (nom, device, M, N) -> variante par défaut (bornée)
        self._params = {}         # (arg, valeurs, device) -> tenseur paramètre (bornée)
        self._shapes = {}         # M -> formes et dtypes Torch des sorties (bornée)
        self._grads = {}          # frozenset(args) -> (GradientPlans, Reduction i, Reduction j)
        self._batched = {}        # rang de batch -> Reduction du plan batch_rank=r (avenant v0.6)
        self._layouts = {}        # (device, B, formes, strides) -> BatchLayout (bornée)
        self._ranged = None       # Reduction du plan ranges=True (avenant v0.6 §2)
        self._range_layouts = {}  # identité des tenseurs de ranges, tile_m -> RangesLayout (bornée)
        self._dt = None
        if backend != "auto":
            get_backend(backend)  # erreur explicite dès compile() si absent

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
                f"backend {name!r} : fp_contract=False (aucune contraction FMA) non pris en charge "
                "par ce backend (contrat v0.3)")
        support = backend.supports(self.plan, device)
        if not support.ok:
            raise UnsupportedError(f"backend {name!r} : {support.reason}")
        scheme = self.plan.sum_scheme
        if (scheme != "direct" and self.plan.reduction.kind in _SUMMING_KINDS
                and scheme not in getattr(backend, "sum_schemes", ("direct",))):
            raise UnsupportedError(
                f"backend {name!r} : sum_scheme={scheme!r} non pris en charge (avenant v0.7, "
                "aucun repli sur 'direct')")
        if self.plan.ranges and not getattr(backend, "ranges", False):
            raise UnsupportedError(
                f"backend {name!r} : ranges block-sparse (avenant v0.6) non pris en charge")
        if self.plan.batch_rank and not getattr(backend, "batch", False):
            raise UnsupportedError(
                f"backend {name!r} : dimensions de batch (avenant v0.6) non prises en charge")
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
        """Variante et objet compilé pour cet appel, selon ``self.tuning``.

        Wisdom v3 : ``nb`` (batch) et ``density()`` (densité des ranges, calculée
        seulement hors ``estimate`` : synchronisation hôte) entrent dans la classe
        de taille ; ``launch_kwargs(compiled)`` complète les lancements mesurés.
        """
        name, backend = self._resolve(device)
        has_variants = self._resolved[str(device)][2]
        level, plan = self.tuning, self.plan
        rho = density() if (density is not None and level != "estimate") else 1.0
        sclass = _tuning.size_class(m, n, nb, rho)
        if not has_variants:
            if level != "estimate" and name not in self._warned:
                self._warned.add(name)
                _tuning.log.warning("backend %r sans variantes : tuning=%r retombe sur 'estimate'", name, level)
            reason = "backend sans variantes" if level != "estimate" else ""
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
                decision = _tuning.TuningDecision(level, "estimate", default.as_json(), sclass, reason="pas de wisdom")
            else:
                cands, truncated = _tuning.candidates(backend, plan, device, sizes, level, default)
                variant, results, stable, seconds = _tuning.measure(
                    level, lambda v: self._compile(name, backend, device, v), cands, tensors, new_outputs, device,
                    launch_kwargs=launch_kwargs)
                decision = _tuning.TuningDecision(
                    level, "measured", variant.as_json(), sclass, stable=stable, candidates=len(cands),
                    truncated_from=truncated, measurements=results, tuning_seconds=seconds,
                    budget_seconds=_tuning.BUDGET_SECONDS[level],
                    reason="" if stable else "dispersion forte : non persistée")
                if stable:
                    _tuning.store(plan.key(), name, identity, sclass, variant, {
                        "level": level, "median_ms": min(r["median_ms"] for r in results),
                        "candidates": len(results), "created": time.time()})
                    decision.persisted = True
                else:
                    _tuning.log.warning("tuning=%r : mesure instable, décision non persistée", level)
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
            raise ValidationError(f"arguments manquants {missing}, inattendus {unexpected}")
        dt = self._dt
        if dt is None:
            dt = self._dt = torch_dtype(plan.dtype)
        Tensor = torch.Tensor

        device = None
        for a in plan.args:
            if a.role != "param":
                v = kwargs[a.name]
                if not isinstance(v, Tensor):
                    raise ValidationError(f"{a.name!r} : torch.Tensor (L, {a.dim}) attendu, reçu {type(v).__name__}")
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
                raise ValidationError(f"{a.name!r} : torch.Tensor (L, {a.dim}) attendu, reçu {type(v).__name__}")
            if v.dtype != dt:
                raise ValidationError(
                    f"{a.name!r} : dtype {v.dtype} au lieu de {dt} (aucune conversion implicite)"
                )
            if v.device != device:
                raise ValidationError(f"{a.name!r} : device {v.device} différent de {device}")
            shape = v.shape
            if role == "param":
                if len(shape) == 0 and a.dim == 1:
                    v = v.reshape(1)
                    shape = v.shape
                if len(shape) != 1 or shape[0] != a.dim:
                    raise ValidationError(f"{a.name!r} : forme {tuple(shape)} au lieu de ({a.dim},)")
            else:
                if len(shape) != 2 or shape[1] != a.dim:
                    raise ValidationError(
                        f"{a.name!r} : forme {tuple(shape)} au lieu de (L, {a.dim}) (rôle {role})"
                    )
                length = shape[0]
                if role == "i":
                    if m is not None and m != length:
                        raise ValidationError(f"{a.name!r} : M={length} incohérent avec M={m}")
                    m = length
                else:
                    if n is not None and n != length:
                        raise ValidationError(f"{a.name!r} : N={length} incohérent avec N={n}")
                    n = length
            if not v.is_contiguous():
                raise ValidationError(f"{a.name!r} : tenseur non contigu refusé (appeler .contiguous())")
            tensors[a.name] = v
        return tensors, device, m, n, None

    def _validate_batched(self, kwargs: dict, device, r: int):
        """Validation des entrées avec dimensions de batch (avenant v0.6)."""
        import torch

        plan = self.plan
        if r < 1 or (plan.batch_rank and r != plan.batch_rank):
            first = next(a for a in plan.args if a.role != "param")
            want = f"(*B, L, {first.dim})" if plan.batch_rank else f"(L, {first.dim}) ou (*B, L, {first.dim})"
            raise ValidationError(f"{first.name!r} : forme {tuple(kwargs[first.name].shape)} au lieu de {want}")
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
                raise ValidationError(f"{a.name!r} : torch.Tensor (*B, L, {a.dim}) attendu, reçu {type(v).__name__}")
            if v.dtype != dt:
                raise ValidationError(f"{a.name!r} : dtype {v.dtype} au lieu de {dt} (aucune conversion implicite)")
            if v.device != device:
                raise ValidationError(f"{a.name!r} : device {v.device} différent de {device}")
            shape, st = tuple(v.shape), v.stride()
            if role == "param":
                if len(shape) == 0 and a.dim == 1:
                    v = v.reshape(1)
                    shape, st = (1,), (1,)
                if len(shape) not in (1, r + 1) or shape[-1] != a.dim:
                    raise ValidationError(
                        f"{a.name!r} : forme {shape} au lieu de ({a.dim},) ou (*B, {a.dim}) "
                        f"avec {r} dimension(s) de batch")
                if a.dim > 1 and st[-1] != 1:
                    raise ValidationError(f"{a.name!r} : dernière dimension non contiguë refusée")
                if len(shape) > 1:
                    batch_shapes.append(shape[:-1])
            else:
                if len(shape) != r + 2 or shape[-1] != a.dim:
                    raise ValidationError(
                        f"{a.name!r} : forme {shape} au lieu de (*B, L, {a.dim}) avec {r} dimension(s) de "
                        f"batch (rang commun à toutes les entrées i et j, rôle {role})")
                length = shape[-2]
                if role == "i":
                    if m is not None and m != length:
                        raise ValidationError(f"{a.name!r} : M={length} incohérent avec M={m}")
                    m = length
                else:
                    if n is not None and n != length:
                        raise ValidationError(f"{a.name!r} : N={length} incohérent avec N={n}")
                    n = length
                if (a.dim > 1 and st[-1] != 1) or (length > 1 and st[-2] != a.dim):
                    raise ValidationError(
                        f"{a.name!r} : deux dernières dimensions non contiguës refusées "
                        "(les dimensions de batch peuvent avoir des strides quelconques)")
                batch_shapes.append(shape[:-2])
            tensors[a.name] = v
        return tensors, device, m, n, _broadcast_batch(tuple(batch_shapes))

    def _param_from_python(self, a, v, dt, device):
        import torch

        values = tuple(v) if isinstance(v, (list, tuple)) else (v,)
        if len(values) != a.dim:
            raise ValidationError(f"{a.name!r} : {a.dim} valeurs attendues, reçu {len(values)}")
        for x in values:
            if isinstance(x, bool) or not isinstance(x, numbers.Real):
                raise ValidationError(f"{a.name!r} : valeur réelle attendue, reçu {x!r}")
        # Tenseur mis en cache par valeurs exactes (type compris) : évite une
        # copie hôte -> device par appel. Les entrées sont en lecture seule
        # pour les backends (contrat), le tenseur partagé n'est jamais modifié.
        key = (a.name, tuple((type(x), x) for x in values), str(device))
        hit = self._params.get(key)
        if hit is not None:
            return hit
        conv = [convert_float(float(x), str(dt).removeprefix("torch."), f"paramètre {a.name}") for x in values]
        t = torch.tensor(conv, dtype=dt, device=device)
        if len(self._params) >= 64:
            self._params.clear()
        self._params[key] = t
        return t

    # ---- exécution
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

    # ---- autograd (cf. kernox.autograd)
    def _forward(self, tensors: dict):
        """Passe avant sur des tenseurs déjà validés (détachés)."""
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
        """Gradients ``{nom: tenseur}`` des entrées ``wrt`` (réductions fusionnées).

        ``ranges`` : ``(triplet avant, triplet transposé)`` dans l'orientation du
        plan (avenant v0.6 §2) ; le côté j des gradients utilise le transposé.
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
                        "gradient par rapport à un argument réduit avec ranges : le triplet transposé "
                        "(ranges_j, slices_j, redranges_i en orientation utilisateur) est requis")
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
                # batch : somme sur les dimensions diffusées de l'entrée (avenant v0.6)
                grads[name] = g.sum_to_size(shape) if batched else g.reshape(shape)
        return grads

    def _backward_surrogate(self, tensors: dict, out, grad_out, wrt, ranges=None):
        """Gradients de LOGSUMEXP/LOGSUMEXPWEIGHT/SUMSOFTMAXWEIGHT (plan SUM substitut)."""
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
        """Gradients de MIN/MAX/KMIN au terme retenu (indices de la passe avant ou recalculés)."""
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
            compiled.launch(tensors, outputs)  # toujours la variante retenue
            self.last_decision = decision
        else:
            # le backend doit exister et accepter le plan même sans lancement
            self._resolve(device)
        if len(outputs) == 2:  # KMIN_ARGKMIN, MIN_ARGMIN, MAX_ARGMAX
            return outputs["value"], outputs["index"]
        return next(iter(outputs.values()))

    # ---- batch (avenant v0.6)
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
        """Tables d'offsets par argument (hôte et device), en cache par formes et strides."""
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
            lead = len(B) - len(bshape)  # param sans batch : bshape vide
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
            self._resolve(device)  # le backend doit exister et accepter le plan batché
        outs = [o.view(*B, *o.shape[1:]) for o in outputs.values()]
        if len(outs) == 2:  # (valeurs, indices)
            return outs[0], outs[1]
        return outs[0]

    # ---- ranges block-sparse (avenant v0.6 §2)
    # ---- opérateur linéaire et solve (parité PyKeOps, cf. kernox.linalg)
    def linear_operator(self, var: str, /, **inputs):
        """:class:`~kernox.linalg.LinearOperator` ``v ↦ self(**inputs, var=v)`` (SUM linear in ``var``)."""
        from .linalg import LinearOperator

        return LinearOperator(self, var, inputs)

    def solve(self, var: str, rhs, /, *, alpha: float = 1e-10, eps: float = 1e-6, maxiter=None, x0=None,
              inputs: dict | None = None, **kwargs):
        """Solves ``(alpha·I + K) var = rhs`` by conjugate gradient, with ``K = self`` viewed as linear in ``var``.

        The other inputs are passed by keyword (or ``inputs={...}`` if a
        name conflicts with ``alpha``, ``eps``, ``maxiter``, ``x0``).
        See :meth:`kernox.linalg.LinearOperator.solve`.
        """
        merged = dict(inputs or {})
        clash = merged.keys() & kwargs.keys()
        if clash:
            raise ValidationError(f"solve : entrées données deux fois {sorted(clash)}")
        merged.update(kwargs)
        return self.linear_operator(var, **merged).solve(rhs, alpha=alpha, eps=eps, maxiter=maxiter, x0=x0)

    def with_ranges(self, ranges, *, validate: bool = False) -> "RangedReduction":
        """Call restricted to the blocks of ``ranges`` (:class:`kernox.Ranges`).

        ``run.with_ranges(r)(**inputs)``. ``validate=True`` checks the values
        (bounds, overlaps; one host synchronization per call).
        """
        from .ranges import Ranges

        if not isinstance(ranges, Ranges):
            raise ValidationError(f"with_ranges attend un kernox.Ranges, reçu {type(ranges).__name__}")
        fwd, bwd = ranges.oriented(bool(self.plan.meta.get("swapped_ij")))
        return RangedReduction(self, fwd, bwd, validate=validate)

    def _run_ranged(self, tensors, device, m, n, triplet):
        child = self._ranged
        if child is None:
            child = self._ranged = Reduction(dataclasses.replace(self.plan, ranges=True), self.backend, self.tuning)
        return child._launch_ranged(tensors, device, m, n, triplet)

    def _ranges_layout(self, triplet, device, tile_m, m=None, n=None) -> RangesLayout:
        """Layout des ranges (en cache par identité des tenseurs) ; avec ``m`` et ``n``,
        contrôle des bornes **toujours actif** : une tuile ou un intervalle hors de
        ``[0, m] × [0, n]`` (ou un indice d'intervalle hors de ``redranges``) ferait
        lire ou écrire le noyau hors des tenseurs, ou boucler sur des milliards de j
        (perte du device). Résumé calculé une fois sur les copies hôte du layout,
        coût O(1) par appel ; ``validate=True`` ajoute les recouvrements."""
        layout = self._ranges_layout_cached(triplet, device, tile_m)
        if m is not None:
            s = layout.summary
            kok = s.nw == 0 or (s.k0_min >= 0 and s.dk_min >= 0 and s.k1_max <= s.nr)
            if not kok or s.live_i0_min < 0 or s.live_i1_max > m or s.live_c_min < 0 or s.live_d_max > n:
                raise ValidationError(
                    f"ranges hors bornes pour M={m}, N={n} : lignes [{s.live_i0_min}, {s.live_i1_max}), "
                    f"j [{s.live_c_min}, {s.live_d_max})"
                    + ("" if kok else ", slices hors de redranges ou décroissantes"))
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
        layout = RangesLayout(work, red, tile_m)  # copies hôte à la demande (backends de référence)
        if len(self._range_layouts) >= 64:
            self._range_layouts.clear()
        self._range_layouts[key] = (triplet.rows, triplet.red, layout)
        layout.summary  # noqa: B018 - calculé une fois, hors du chemin chaud
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
    """Fraction ρ des paires (i, j) réduites par le triplet (wisdom v3 ; synchronisation hôte)."""
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
    """Entrées complexes vues en réel entrelacé, sortie SUM complexe rendue complexe."""
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
        for tr, rows, cols, side in ((self._fwd, m, n, "avant"), (self._bwd, n, m, "transposé")):
            if tr is None:
                continue
            if tr.rows.device != device:
                raise ValidationError(f"ranges ({side}) : device {tr.rows.device} différent de {device}")
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
            raise UnsupportedError("ranges et dimensions de batch combinés : refusé (avenant v0.6)")
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
            fast_math: bool | None = None) -> Reduction:
    """Compiles a frontend reduction (or a :class:`Plan`) into a callable.

    ``dtype`` is mandatory (``"float16"``, ``"float32"`` or ``"float64"``): no
    implicit precision. ``backend``: ``"auto"`` (device of the tensors:
    cuda/hip/sycl), a native name, or ``"oracle"`` (explicit reference).
    ``tuning``: ``"estimate"`` (default, no measurement), ``"wisdom_only"``,
    ``"measure"`` or ``"patient"`` (see :mod:`kernox.tuning`).
    ``fp_contract``: ``True`` by default, FMA contraction
    allowed within an expression; ``False`` forbids it on all
    backends (bit-for-bit reproducibility; a backend that cannot
    guarantee it raises :class:`UnsupportedError`). Part of the plan key.
    ``sum_scheme``: ``"direct"`` (default), ``"block"`` or
    ``"kahan"`` (Neumaier) for SUM, LOGSUMEXP and SUMSOFTMAXWEIGHT; ignored by
    the other reductions. A backend that does not implement it raises
    :class:`UnsupportedError`, never a fallback to ``"direct"``.
    ``fast_math``: ``False`` by default (IEEE); ``True`` allows
    approximate exp/log (CUDA FP32: ``ex2.approx``/``lg2.approx``, like PyKeOps
    ``use_fast_math``). Part of the plan key; a backend that does not
    implement it raises :class:`UnsupportedError`. The oracle stays IEEE.
    """
    from .expr import ReductionExpr

    if fast_math is not None and not isinstance(fast_math, bool):
        raise ValidationError(f"fast_math doit être un booléen, reçu {fast_math!r}")
    if fp_contract is not None and not isinstance(fp_contract, bool):
        raise ValidationError(f"fp_contract doit être un booléen, reçu {fp_contract!r}")
    if isinstance(expr, ReductionExpr):
        plan = expr.to_plan(dtype, acc_dtype=acc_dtype, fp_contract=True if fp_contract is None else fp_contract,
                            sum_scheme=sum_scheme, fast_math=bool(fast_math))
    elif isinstance(expr, Plan):
        if dtype != expr.dtype or (acc_dtype is not None and acc_dtype != expr.acc_dtype):
            raise ValidationError("dtype/acc_dtype incohérents avec le plan fourni")
        if fp_contract is not None and fp_contract != expr.fp_contract:
            raise ValidationError("fp_contract incohérent avec le plan fourni")
        if fast_math is not None and fast_math != expr.fast_math:
            raise ValidationError("fast_math incohérent avec le plan fourni")
        if sum_scheme is not None and sum_scheme != expr.sum_scheme:
            raise ValidationError("sum_scheme incohérent avec le plan fourni")
        plan = expr
    else:
        raise TypeError(
            "compile attend une réduction (ex. formule.sum_over('j')) ou un Plan, "
            f"reçu {type(expr).__name__}"
        )
    return Reduction(plan, backend, tuning)


__all__ = ["compile", "Reduction", "RangedReduction", "register_backend", "get_backend", "auto_backend_name", "NATIVE_BACKENDS"]
