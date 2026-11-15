"""Clé stable d'un :class:`kernox.plan.Plan` (contrat v0).

La clé est le sha256 hexadécimal d'une sérialisation JSON canonique :

- version du format de clé (``PLAN_KEY_FORMAT``) ;
- pour chaque argument, dans l'ordre de ``plan.args`` : nom, rôle, dimension
  et ``srepr`` symox de chaque symbole de composante (nom + hypothèses) ;
- ``srepr`` symox de chaque sortie, dans l'ordre ; si l'arbre déplié d'une
  sortie dépasse ``DAG_THRESHOLD`` nœuds (DAG profond partagé : ``srepr`` serait
  exponentiel), la sortie est sérialisée comme graphe : ``["dag", nœuds]``, la
  liste des nœuds distincts en ordre postfixe, chacun ``srepr`` pour un atome
  ou ``[classe, tête, indices des enfants]`` sinon. Les clés des plans
  existants (arbres plus petits) sont inchangées ;
- réduction (type, K), ``dtype``, ``acc_dtype``, ``index_dtype``, ``fast_math``, ``fp_contract`` ;
- ``sum_scheme`` (avenant v0.7) seulement s'il diffère de ``"direct"`` : la
  clé d'un plan existant est inchangée.
- ``batch_rank`` (avenant v0.6) seulement s'il est non nul : la clé d'un plan
  sans batch est inchangée ;
- ``ranges`` (avenant v0.6 §2) seulement s'il est vrai.

``plan.meta`` n'entre pas dans la clé (informations de diagnostic seulement).
La clé ne dépend ni de ``PYTHONHASHSEED`` ni de l'ordre de création des objets
(testé dans ``tests/test_plan_key.py``). Les backends y ajoutent la version du
générateur, le compilateur, les options et l'architecture (cf. contrat).
"""

from __future__ import annotations

import hashlib
import json

PLAN_KEY_FORMAT = "kernox-plan-key-v1"


#: taille d'arbre déplié au-delà de laquelle une sortie est sérialisée comme graphe
DAG_THRESHOLD = 200_000


def _srepr(expr) -> str:
    from symox import srepr

    return srepr(expr)


def _tree_size(expr, limit):
    """Nœuds de l'arbre déplié (mémoïsé par sous-expression), plafonné à ``limit``."""
    memo = {}

    def size(e):
        hit = memo.get(e)
        if hit is None:
            hit = min(limit, 1 + sum(size(a) for a in e.args))
            memo[e] = hit
        return hit

    return size(expr)


def _dag(expr) -> list:
    """Sérialisation canonique du graphe : nœuds distincts en ordre postfixe."""
    from symox import srepr

    index, nodes = {}, []

    def visit(e):
        hit = index.get(e)
        if hit is not None:
            return hit
        if not e.args:
            node = srepr(e)
        else:
            children = [visit(a) for a in e.args]
            node = [type(e).__name__, str(e.func), children]
        index[e] = len(nodes)
        nodes.append(node)
        return index[e]

    visit(expr)
    return ["dag", nodes]


def _output(expr):
    if _tree_size(expr, DAG_THRESHOLD + 1) > DAG_THRESHOLD:
        return _dag(expr)
    return _srepr(expr)


def plan_payload(plan) -> dict:
    """Structure JSON canonique hachée par :func:`plan_key` (utile au débogage)."""
    payload = {
        "format": PLAN_KEY_FORMAT,
        "args": [
            [a.name, a.role, int(a.dim), [_srepr(s) for s in a.symbols]]
            for a in plan.args
        ],
        "outputs": [_output(e) for e in plan.outputs],
        "reduction": [plan.reduction.kind.value, plan.reduction.k],
        "dtype": plan.dtype,
        "acc_dtype": plan.acc_dtype,
        "index_dtype": plan.index_dtype,
        "fast_math": bool(plan.fast_math),
        "fp_contract": bool(plan.fp_contract),
    }
    scheme = getattr(plan, "sum_scheme", "direct")
    if scheme != "direct":
        payload["sum_scheme"] = scheme
    rank = int(getattr(plan, "batch_rank", 0))
    if rank:
        payload["batch_rank"] = rank
    if getattr(plan, "ranges", False):
        payload["ranges"] = True
    return payload


def plan_key(plan) -> str:
    """sha256 hex de :func:`plan_payload`, sérialisé de façon canonique."""
    text = json.dumps(plan_payload(plan), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("ascii")).hexdigest()
