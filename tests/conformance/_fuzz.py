"""Générateur de formules aléatoires reproductible par seed, et minimisation.

Une formule est un arbre de tuples ``(op, *enfants)`` ; ``source(tree)`` en
donne le texte Python du frontend (``kp.*``), qui sert aussi d'enregistrement
de régression : il est réévalué dans un espace de noms restreint.
Couverture : + − × /, moins unaire, fonctions usuelles, puissances entières et
rationnelles, ``where`` imbriqués (domaines gardés compris), constantes
Python, comparaisons et opérateurs logiques. Les fonctions à domaine restreint
reçoivent un argument rendu valide (``abs(e) + 0.5``) ou sont gardées par
``where`` ; ``exp`` reçoit un argument borné.
"""

from __future__ import annotations

import random

import numpy as np

import kernox as kp
from kernox.plan import ReductionKind as R

CONSTS = (0, 1, 2, 3, -1, 0.5, 0.25, -1.5, 2.75, 1e-3, 10.0)
UNARY = ("neg", "abs", "sin", "cos", "tanh", "atan", "exp_b", "log_s", "sqrt_s", "rsqrt_s",
         "pow2", "pow3", "powm1_s", "powh_s", "pow32_s", "powmh_s", "square", "clamp", "floor")
BINARY = ("add", "sub", "mul", "div_s", "min", "max", "atan2")
COMPARE = ("lt", "le", "gt", "ge", "eq", "ne")
KINDS = (R.SUM, R.MIN, R.MAX, R.ARGMIN, R.ARGMAX, R.KMIN, R.ARGKMIN, R.KMIN_ARGKMIN, R.COUNT)


class Generator:
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)
        r = self.rng
        self.dims = {"x": r.choice((1, 2, 3)), "y": r.choice((1, 2, 3)), "p": r.choice((1, 2))}

    def leaf(self):
        r = self.rng
        u = r.random()
        if u < 0.35:
            return ("var", "x", r.randrange(self.dims["x"]))
        if u < 0.7:
            return ("var", "y", r.randrange(self.dims["y"]))
        if u < 0.8:
            return ("var", "p", r.randrange(self.dims["p"]))
        return ("const", r.choice(CONSTS))

    def real(self, depth):
        r = self.rng
        if depth <= 0 or r.random() < 0.2:
            return self.leaf()
        u = r.random()
        if u < 0.4:
            return (r.choice(UNARY), self.real(depth - 1))
        if u < 0.8:
            return (r.choice(BINARY), self.real(depth - 1), self.real(depth - 1))
        if u < 0.9:
            return ("where", self.cond(depth - 1), self.real(depth - 1), self.real(depth - 1))
        # domaine invalide exclu par where (branche inactive non évaluée)
        g = r.choice(("g_log", "g_sqrt", "g_div"))
        return (g, self.real(depth - 1), self.real(depth - 1))

    def cond(self, depth):
        r = self.rng
        u = r.random()
        if depth <= 0 or u < 0.6:
            return (r.choice(COMPARE), self.real(depth - 1), self.real(depth - 1))
        if u < 0.85:
            return (r.choice(("and", "or", "xor")), self.cond(depth - 1), self.cond(depth - 1))
        return ("not", self.cond(depth - 1))

    def case(self):
        """``dict`` décrivant un cas : arbre(s), réduction, dtype, tailles, données.

        Le contrat v0 exige une dépendance en i et en j : un tirage dont la formule
        (après simplification symox) n'en dépend pas est refait, de façon déterministe.
        """
        for _ in range(100):
            case = self._draw()
            if depends_on_both(case):
                return case
        raise RuntimeError(f"seed {self.seed} : aucune formule dépendant de x et y")

    def _draw(self):
        r = self.rng
        kind = r.choice(KINDS)
        depth = r.choice((2, 3, 4))
        trees = [self.real(depth)]
        if kind in (R.SUM, R.MIN, R.MAX, R.ARGMIN, R.ARGMAX) and r.random() < 0.3:
            trees.append(self.real(depth))  # sortie vectorielle
        if kind is R.COUNT:
            trees = [(r.choice(COMPARE[:4]), self.real(depth), ("const", r.choice(CONSTS)))]
        return {
            "seed": self.seed, "kind": kind.value, "k": r.choice((1, 2, 5, 32, 33)),
            "over": "i" if r.random() < 0.2 else "j",
            "dtype": r.choice(("float32", "float64")),
            "m": r.choice((1, 3, 17, 40)), "n": r.choice((1, 7, 33, 300)),
            "data": r.choice(("normal", "normal", "int")), "dims": dict(self.dims),
            "trees": trees,
        }


# ------------------------------------------------------------------ texte
_UN = {
    "neg": "(-{0})", "abs": "kp.abs({0})", "sin": "kp.sin({0})", "cos": "kp.cos({0})",
    "tanh": "kp.tanh({0})", "atan": "kp.atan({0})", "exp_b": "kp.exp(kp.sin({0}) * 3)",
    "log_s": "kp.log(kp.abs({0}) + 0.5)", "sqrt_s": "kp.sqrt(kp.abs({0}) + 0.25)",
    "rsqrt_s": "kp.rsqrt({0} * {0} + 0.5)", "pow2": "({0}) ** 2", "pow3": "({0}) ** 3",
    "powm1_s": "({0} * {0} + 0.5) ** -1", "powh_s": "(kp.abs({0}) + 0.5) ** 0.5",
    "pow32_s": "(kp.abs({0}) + 0.25) ** 1.5", "powmh_s": "({0} * {0} + 1) ** -0.5",
    "square": "kp.square({0})", "clamp": "kp.clamp({0}, -1, 1.5)", "floor": "kp.floor({0})",
}
_BIN = {
    "add": "({0} + {1})", "sub": "({0} - {1})", "mul": "({0} * {1})",
    "div_s": "({0} / ({1} * {1} + 0.5))", "min": "kp.minimum({0}, {1})", "max": "kp.maximum({0}, {1})",
    "atan2": "kp.atan2({0}, {1})",
    "g_log": "kp.where({0} > 0, kp.log({0}), {1})", "g_sqrt": "kp.where({0} >= 0, kp.sqrt({0}), {1})",
    "g_div": "kp.where(kp.ne({0}, 0), {1} / {0}, 0)",
    "lt": "({0} < {1})", "le": "({0} <= {1})", "gt": "({0} > {1})", "ge": "({0} >= {1})",
    "eq": "kp.eq({0}, {1})", "ne": "kp.ne({0}, {1})",
    "and": "({0} & {1})", "or": "({0} | {1})", "xor": "({0} ^ {1})",
}


def source(t) -> str:
    op = t[0]
    if op == "var":
        return f"{t[1]}[{t[2]}]"
    if op == "const":
        return repr(t[1])
    if op == "not":
        return f"(~{source(t[1])})"
    if op == "where":
        return f"kp.where({source(t[1])}, {source(t[2])}, {source(t[3])})"
    if op in _UN:
        return _UN[op].format(source(t[1]))
    return _BIN[op].format(*(source(c) for c in t[1:]))


def _formula(case):
    dims = case["dims"]
    ns = {"kp": kp, "x": kp.Vi("x", dims["x"]), "y": kp.Vj("y", dims["y"]), "p": kp.Param("p", dims["p"])}
    texts = case.get("sources") or [source(t) for t in case["trees"]]
    parts = [eval(s, {"__builtins__": {}}, ns) for s in texts]  # noqa: S307 - textes générés ici
    f = parts[0] if len(parts) == 1 else kp.cat(*parts)
    return f, texts


def depends_on_both(case) -> bool:
    try:
        f, _ = _formula(case)
    except Exception:  # noqa: BLE001 - tirage invalide : refait
        return False
    names = set(getattr(f, "_vars", {}))
    return {"x", "y"} <= names


def build(case: dict):
    """Expression de réduction du frontend décrite par ``case`` (textes ou arbres)."""
    f, texts = _formula(case)
    kind, over, k = R(case["kind"]), case["over"], case["k"]
    return {
        R.SUM: lambda: f.sum_over(over), R.COUNT: lambda: f.sum_over(over),
        R.MIN: lambda: f.min(over=over), R.MAX: lambda: f.max(over=over),
        R.ARGMIN: lambda: f.argmin(over=over), R.ARGMAX: lambda: f.argmax(over=over),
        R.KMIN: lambda: f.kmin(k, over=over), R.ARGKMIN: lambda: f.argkmin(k, over=over),
        R.KMIN_ARGKMIN: lambda: f.kmin_argkmin(k, over=over),
    }[kind](), texts


def arrays(case: dict):
    rng = np.random.default_rng([case["seed"], case["m"], case["n"]])
    dims = case["dims"]
    rows_x, rows_y = (case["m"], case["n"]) if case["over"] == "j" else (case["n"], case["m"])
    if case["data"] == "int":
        gen = lambda shape: rng.integers(-3, 4, shape).astype(np.float64)  # noqa: E731
    else:
        gen = lambda shape: rng.standard_normal(shape)  # noqa: E731
    return {"x": gen((rows_x, dims["x"])), "y": gen((rows_y, dims["y"])), "p": np.abs(gen(dims["p"])) + 0.5}


# ---------------------------------------------------------------- minimisation
def _subtrees(t):
    """Remplacements candidats d'un nœud, du plus simple au plus complexe."""
    out = []
    if t[0] not in ("var", "const"):
        out += [c for c in t[1:] if isinstance(c, tuple) and c[0] not in COMPARE + ("and", "or", "xor", "not")]
        out += [("var", "x", 0), ("var", "y", 0), ("const", 1)]
    return out


def _replace(t, path, new):
    if not path:
        return new
    i = path[0]
    return t[:i] + (_replace(t[i], path[1:], new),) + t[i + 1:]


def _paths(t, prefix=()):
    yield prefix
    if t[0] in ("var", "const"):
        return
    for i, c in enumerate(t[1:], start=1):
        if isinstance(c, tuple):
            yield from _paths(c, prefix + (i,))


def _get(t, path):
    for i in path:
        t = t[i]
    return t


def minimize(case: dict, fails, budget: int = 80) -> dict:
    """Réduit l'arbre et les tailles tant que ``fails(case)`` reste vrai."""
    best = dict(case)
    tries = 0
    changed = True
    while changed and tries < budget:
        changed = False
        for size_key in ("m", "n"):
            while best[size_key] > 1 and tries < budget:
                cand = dict(best, **{size_key: max(1, best[size_key] // 2)})
                tries += 1
                if fails(cand):
                    best, changed = cand, True
                else:
                    break
        for ti, tree in enumerate(best["trees"]):
            for path in list(_paths(tree)):
                if tries >= budget:
                    break
                try:
                    node = _get(best["trees"][ti], path)
                except (IndexError, TypeError):
                    continue
                if not isinstance(node, tuple):
                    continue
                for new in _subtrees(node):
                    cand_trees = list(best["trees"])
                    cand_trees[ti] = _replace(cand_trees[ti], path, new)
                    cand = dict(best, trees=cand_trees)
                    tries += 1
                    if not depends_on_both(cand):
                        continue
                    if fails(cand):
                        best, changed = cand, True
                        break
    return best
