import os
import subprocess
import sys
import textwrap

from kernox import Vi, Vj, Param, exp


def _plan(dtype="float32"):
    x, y, s = Vi("x", 3), Vj("y", 3), Param("s")
    return (exp(-((x - y) ** 2).sum() / s) * 2.0).sum_over("j").to_plan(dtype)


def test_key_is_stable_and_discriminant():
    p1, p2 = _plan(), _plan()
    assert p1.key() == p2.key() and len(p1.key()) == 64
    assert _plan("float64").key() != p1.key()
    x, y = Vi("x", 3), Vj("y", 3)
    assert ((x - y) ** 2).sum().sum_over("j").to_plan("float32").key() != \
        ((x - y) ** 2).sum().min(over="j").to_plan("float32").key()
    assert ((x - y) ** 2).sum().kmin(3, over="j").to_plan("float32").key() != \
        ((x - y) ** 2).sum().kmin(4, over="j").to_plan("float32").key()


def test_key_meta_excluded():
    from dataclasses import replace
    p = _plan()
    assert replace(p, meta={"foo": 1}).key() == p.key()


def test_key_independent_of_hash_seed():
    code = textwrap.dedent("""
        from kernox import Vi, Vj, Param, exp, where
        x, y, s = Vi("x", 3), Vj("y", 3), Param("s")
        d2 = ((x - y) ** 2).sum()
        print(where(d2 < s, exp(-d2) + 0.1, 0).sum_over("j").to_plan("float32").key())
    """)
    keys = set()
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        keys.add(subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip())
    assert len(keys) == 1


def test_deep_shared_outputs_get_a_graph_key():
    """Arbre déplié énorme (DAG partagé) : clé sérialisée comme graphe, rapide,
    stable et distincte d'un graphe voisin ; les petits plans gardent ``srepr``."""
    import time

    import kernox as kp
    from kernox import _plan_key

    x, y = kp.Vi("x", 1), kp.Vj("y", 1)

    def deep(c):
        e = x
        for _ in range(40):
            e = kp.exp(e - y) * c + kp.sin(e) + x
        return e.sum_over("j").to_plan("float64")

    start = time.perf_counter()
    p = deep(0.5)
    payload = _plan_key.plan_payload(p)
    assert payload["outputs"][0][0] == "dag"
    assert p.key() == deep(0.5).key() != deep(0.25).key()
    assert time.perf_counter() - start < 60
    small = (x * y).sum_over("j").to_plan("float64")
    assert isinstance(_plan_key.plan_payload(small)["outputs"][0], str)
