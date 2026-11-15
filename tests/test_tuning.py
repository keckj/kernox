"""Autotuning (v0.2 contract) with a fake multi-variant backend, on CPU."""
import json
import logging
import threading

import pytest

torch = pytest.importorskip("torch")

import kernox as kp
from kernox import Vi, Vj, tuning
from kernox.backends.base import CompileInfo, Support, Variant
from kernox.errors import ValidationError
from kernox.oracle import OracleReduction
from kernox.runtime import register_backend


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNOX_CACHE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def simulated_timer(monkeypatch):
    """Durations simulated by the fake backend: deterministic tests even on a loaded machine.

    The real timer (GPU events) is covered by ``test_gpu_event_timer_on_current_stream``.
    """
    def run(self, fn):
        fn()
        return FakeCompiled.last_ms

    monkeypatch.setattr(tuning._Timer, "run", run)


class FakeCompiled:
    last_ms = 0.0

    def __init__(self, backend, plan, variant):
        self.backend, self.variant = backend, variant
        self.ref = OracleReduction(plan, "cpu")
        self.info = CompileInfo("fakev", "cpu", "fake-arch", backend.compiler, (), False, 0.0, 0, "")

    ranges_tile_m = 1024

    def launch(self, inputs, outputs, batch=None, ranges=None):
        index = int(self.variant.name[1:])
        self.backend.launches.append(self.variant.name)
        FakeCompiled.last_ms = 1e3 * self.backend.delay(index)
        self.ref.launch(inputs, outputs, batch=batch, ranges=ranges)
        # tags the variant in the result (fake backend: semantics are irrelevant)
        outputs["value"] += 1000 * index


class FakeVariantBackend:
    name = "fakev"
    batch = True    # v0.6 amendment: delegated to the oracle
    ranges = True

    def __init__(self, delays=(0.012, 0.003, 0.006), n_variants=None, jitter=False):
        self.delays, self.jitter = list(delays), jitter
        self.n_variants = n_variants or len(delays)
        self.compiler = "fakecc 1.0"
        self.launches, self.compiles, self._tick = [], [], 0

    def delay(self, index):
        base = self.delays[index % len(self.delays)]
        if self.jitter:
            self._tick += 1
            return base * (5 if self._tick % 2 else 1)
        return base

    def supports(self, plan, device):
        return Support(True)

    def default_variant(self, plan, device, sizes):
        return Variant("v0")

    def variants(self, plan, device, sizes, effort):
        return [Variant(f"v{i}", (("block", 64 * (i + 1)),)) if i else Variant("v0") for i in range(self.n_variants)]

    def device_identity(self, device):
        return {"device_name": "fake", "compiler": self.compiler}

    def compile(self, plan, device, variant=None):
        variant = variant or Variant("v0")
        self.compiles.append(variant.name)
        return FakeCompiled(self, plan, variant)


def install(**kw):
    backend = FakeVariantBackend(**kw)
    register_backend("fakev", lambda: backend, replace=True)
    return backend


def reduction(tuning_level, backend="fakev"):
    x, y = Vi("x", 2), Vj("y", 2)
    return kp.compile(((x - y) ** 2).sum().sum_over("j"), dtype="float64", backend=backend, tuning=tuning_level)


def data(m=8, n=16):
    g = torch.Generator().manual_seed(0)
    return dict(x=torch.randn(m, 2, generator=g, dtype=torch.float64),
                y=torch.randn(n, 2, generator=g, dtype=torch.float64))


def exact(d):
    return ((d["x"][:, None, :] - d["y"][None, :, :]) ** 2).sum(-1).sum(-1, keepdim=True)


def test_level_validation():
    install()
    with pytest.raises(ValidationError):
        reduction("fast")


def test_estimate_never_measures_nor_reads_wisdom():
    fake = install()
    d = data()
    out = reduction("estimate")(**d)
    assert fake.launches == ["v0"]
    torch.testing.assert_close(out, exact(d))
    reduction("measure")(**d)  # actual wisdom: v1
    assert tuning.export_wisdom()["devices"]
    run = reduction("estimate")
    n = len(fake.launches)
    run(**d)
    assert run.last_decision.source == "estimate" and fake.launches[n:] == ["v0"]


def test_measure_picks_fastest_persists_and_returns_its_result():
    fake = install()
    d = data()
    run = reduction("measure")
    out = run(**d)
    dec = run.last_decision
    assert dec.source == "measured" and dec.variant["name"] == "v1" and dec.stable and dec.persisted
    assert dec.candidates == 3 and len(dec.measurements) == 3
    assert fake.launches[-1] == "v1"  # last launch: selected variant, real outputs
    torch.testing.assert_close(out, exact(d) + 1000)  # result produced by v1
    # same size class: no new measurement, no recompilation
    n_launch, n_compile = len(fake.launches), len(fake.compiles)
    run(**data(m=7, n=15))
    assert len(fake.launches) == n_launch + 1 and len(fake.compiles) == n_compile
    # new object: wisdom read back, no measurement
    run2 = reduction("measure")
    run2(**d)
    assert run2.last_decision.source == "wisdom" and fake.launches[-1] == "v1"
    assert len(fake.launches) == n_launch + 2
    # wisdom_only reads the same entry; another size class without wisdom -> estimate
    w = reduction("wisdom_only")
    w(**d)
    assert w.last_decision.source == "wisdom"
    w(**data(m=100, n=16))
    assert w.last_decision.source == "estimate" and fake.launches[-1] == "v0"


def test_new_size_class_measures_again():
    fake = install()
    run = reduction("measure")
    run(**data(8, 16))
    before = len(fake.launches)
    run(**data(64, 16))
    assert len(fake.launches) > before + 1
    assert len(run.tuning_decisions) == 2
    assert {d.size_class for d in run.tuning_decisions} == {(6, 8, 0, 0), (12, 8, 0, 0)}


def test_half_octave_classes():
    from kernox.tuning import size_class

    assert size_class(1000, 1000) == size_class(1024, 1000) != size_class(1100, 1000)
    assert size_class(0, 1) == (0, 0, 0, 0)
    # wisdom v3: nb and ranges density in half-octaves
    assert size_class(8, 16, 1, 1.0) == (6, 8, 0, 0)
    assert size_class(8, 16, 4, 0.25) == (6, 8, 4, 4)
    assert size_class(8, 16, 3, 0.3) == (6, 8, 4, 4) != size_class(8, 16, 3, 0.05)
    assert size_class(8, 16, 1, 0.0) == size_class(8, 16, 1, 1 / 128)  # ρ = 0: one pair


def test_patient_does_not_reuse_measure_wisdom():
    fake = install()
    reduction("measure")(**data())
    w = reduction("wisdom_only")
    w(**data())
    assert w.last_decision.source == "wisdom"
    p = reduction("patient")
    before = len(fake.launches)
    p(**data())
    assert p.last_decision.source == "measured" and len(fake.launches) > before + 1
    m = reduction("measure")
    m(**data())
    assert m.last_decision.source == "wisdom"  # patient entry reusable by measure


def test_version_change_invalidates_wisdom():
    fake = install()
    reduction("measure")(**data())
    fake.compiler = "fakecc 2.0"
    run = reduction("wisdom_only")
    run(**data())
    assert run.last_decision.source == "estimate"
    run = reduction("measure")
    run(**data())
    assert run.last_decision.source == "measured"


def test_unstable_measurement_is_not_persisted(caplog):
    install(jitter=True)
    run = reduction("measure")
    with caplog.at_level(logging.WARNING, logger="kernox.tuning"):
        out = run(**data())
    dec = run.last_decision
    assert dec.source == "measured" and not dec.stable and not dec.persisted
    assert "unstable" in caplog.text
    torch.testing.assert_close(out, exact(data()) + 1000 * int(dec.variant["name"][1:]))
    assert tuning.export_wisdom()["devices"] == {}
    w = reduction("wisdom_only")
    w(**data())
    assert w.last_decision.source == "estimate"


@pytest.mark.parametrize("level,limit", [("measure", 4), ("patient", 32)])
def test_candidate_bound_truncates_and_logs(level, limit, caplog):
    fake = install(delays=(0.0005,), n_variants=40)
    run = reduction(level)
    with caplog.at_level(logging.WARNING, logger="kernox.tuning"):
        run(**data())
    dec = run.last_decision
    assert dec.candidates == limit and dec.truncated_from == 40
    assert len(dec.measurements) <= limit
    assert "truncated" in caplog.text
    assert len(set(fake.compiles)) <= limit


def test_budget_bounds_measurement(monkeypatch):
    install(delays=(0.02,), n_variants=4)
    monkeypatch.setitem(tuning.BUDGET_SECONDS, "measure", 0.0)
    run = reduction("measure")
    run(**data())
    assert len(run.last_decision.measurements) == 1  # at least the default variant


def test_backend_without_variants_falls_back_explicitly(caplog):
    run = reduction("measure", backend="oracle")
    with caplog.at_level(logging.WARNING, logger="kernox.tuning"):
        out = run(**data())
    assert run.last_decision.source == "fallback" and "estimate" in caplog.text
    torch.testing.assert_close(out, exact(data()))


def test_export_import_forget_and_invalid_entries(cache):
    install()
    reduction("measure")(**data())
    exported = tuning.export_wisdom(cache / "w.json")
    assert len(exported["devices"]) == 1
    assert json.loads((cache / "w.json").read_text()) == exported
    assert tuning.forget_wisdom() == 1 and tuning.export_wisdom()["devices"] == {}
    assert tuning.import_wisdom(cache / "w.json") == 1
    run = reduction("wisdom_only")
    run(**data())
    assert run.last_decision.source == "wisdom"
    # tampered entry (inconsistent key) or corrupted file: ignored
    (dev,) = exported["devices"].values()
    key = next(iter(dev["entries"]))
    dev["entries"][key]["size_class"] = [9, 9]
    tuning.forget_wisdom()
    assert tuning.import_wisdom(exported) == 0
    for f in (cache / "wisdom").glob("wisdom-*.json"):
        f.write_text("{corrupted")
    run = reduction("wisdom_only")
    run(**data())
    assert run.last_decision.source == "estimate"
    with pytest.raises(ValueError):
        tuning.import_wisdom({"format": "other"})


def test_concurrent_stores_are_atomic():
    identity = {"device_name": "fake", "compiler": "c", "backend": "b"}

    def worker(i):
        tuning.store(f"plan{i}", "b", identity, (1, 1), Variant("v", (("i", i),)), {})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    (dev,) = tuning.export_wisdom()["devices"].values()
    assert len(dev["entries"]) == 16
    for i in range(16):
        assert tuning.lookup(f"plan{i}", "b", identity, (1, 1)) == Variant("v", (("i", i),))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Torch CUDA/ROCm not available")
def test_gpu_event_timer_on_current_stream():
    dev = torch.device("cuda", torch.cuda.current_device())
    timer = tuning._Timer(dev)
    assert timer.mod is not None
    a = torch.randn(1024, 1024, device=dev)
    ms = timer.run(lambda: a @ a)
    assert ms > 0


def test_batched_plans_are_measured_per_batch_class():
    """Wisdom v3: no more ``fallback`` for batched plans; ``nb`` is part of the class."""
    fake = install()
    run = reduction("measure")
    g = torch.Generator().manual_seed(1)
    x, y = torch.randn(4, 8, 2, generator=g, dtype=torch.float64), torch.randn(4, 16, 2, generator=g, dtype=torch.float64)
    out = run(x=x, y=y)
    dec = run._batched[1].last_decision
    assert dec.source == "measured" and dec.variant["name"] == "v1" and dec.persisted
    assert dec.size_class == (6, 8, 4, 0)
    want = torch.stack([exact(dict(x=x[b], y=y[b])) for b in range(4)])
    torch.testing.assert_close(out, want + 1000)
    # other nb (different class): new measurement; same nb: wisdom read back by a new object
    run(x=x[:1], y=y[:1])
    assert {d.size_class for d in run._batched[1].tuning_decisions} == {(6, 8, 4, 0), (6, 8, 0, 0)}
    run2 = reduction("wisdom_only")
    run2(x=x, y=y)
    assert run2._batched[1].last_decision.source == "wisdom"
    assert fake.launches[-1] == "v1"


def test_ranged_plans_are_measured_per_density_class():
    install()
    run = reduction("measure")
    d = data(m=16, n=16)
    labels = torch.arange(16) // 4
    sparse = run.with_ranges(kp.ranges_from_labels(labels, labels, torch.eye(4, dtype=torch.bool)))
    out = sparse(**d)
    dec = run._ranged.last_decision
    assert dec.source == "measured" and dec.persisted
    assert dec.size_class == (8, 8, 0, 4)  # ρ = 1/4
    mask = (labels[:, None] == labels[None]).to(torch.float64)
    want = (((d["x"][:, None] - d["y"][None]) ** 2).sum(-1) * mask).sum(-1, keepdim=True)
    torch.testing.assert_close(out, want + 1000)
    dense = run.with_ranges(kp.Ranges.dense(16, 16, "cpu"))
    dense(**d)
    assert run._ranged.last_decision.size_class == (8, 8, 0, 0)
    # estimate: the density is never computed (no synchronization)
    est = reduction("estimate").with_ranges(kp.ranges_from_labels(labels, labels, torch.eye(4, dtype=torch.bool)))
    est(**d)
    assert est.red._ranged.last_decision.size_class == (8, 8, 0, 0)
