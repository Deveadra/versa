from __future__ import annotations

import json, re
from typing import Dict, Any, Tuple, List, Callable, Union


def _coerce(val: Any) -> Any:
    """Normalize values from signals into bool/float/int where possible."""
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("true", "false"):
        return s == "true"
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return val


def _get_signal(signals: Dict[str, dict], key: str) -> Any:
    # key may be "signal:name" or just "name"
    name = key.split("signal:", 1)[-1]
    entry = signals.get(name)
    if not entry:
        return None
    return _coerce(entry.get("value"))


def _eval_node(node: Any, signals: Dict[str, dict]) -> Any:
    if isinstance(node, (str, int, float, bool)) or node is None:
        return node
    if isinstance(node, list):
        return [_eval_node(n, signals) for n in node]
    if isinstance(node, dict):
        # JSON-logic style operators
        if "all" in node:
            return all(_eval_node(x, signals) for x in node["all"])
        if "any" in node:
            return any(_eval_node(x, signals) for x in node["any"])
        if "not" in node:
            return not _eval_node(node["not"], signals)
        if "eq" in node:
            a, b = node["eq"]; return _value(a, signals) == _value(b, signals)
        if "neq" in node:
            a, b = node["neq"]; return _value(a, signals) != _value(b, signals)
        if "gt" in node:
            a, b = node["gt"]; return float(_value(a, signals) or 0) > float(_value(b, signals) or 0)
        if "gte" in node:
            a, b = node["gte"]; return float(_value(a, signals) or 0) >= float(_value(b, signals) or 0)
        if "lt" in node:
            a, b = node["lt"]; return float(_value(a, signals) or 0) < float(_value(b, signals) or 0)
        if "lte" in node:
            a, b = node["lte"]; return float(_value(a, signals) or 0) <= float(_value(b, signals) or 0)
        if "exists" in node:
            return _get_signal(signals, node["exists"]) is not None
        if "regex" in node:
            val, pattern = node["regex"]
            return re.search(pattern, str(_value(val, signals) or "")) is not None
        if "between" in node:
            x, lo, hi = node["between"]
            v = float(_value(x, signals) or 0)
            return float(_value(lo, signals) or 0) <= v <= float(_value(hi, signals) or 0)
        # passthrough recursive
        return {k: _eval_node(v, signals) for k, v in node.items()}
    return None


def _value(node: Any, signals: Dict[str, dict]) -> Any:
    if isinstance(node, str) and node.startswith("signal:"):
        return _get_signal(signals, node)
    return _eval_node(node, signals)


def evaluate_condition(condition_json: str, signals: Dict[str, dict]) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Returns (match, severity, bindings).
    - severity: float between 0–1
    - bindings: computed fields for templating
    """
    try:
        obj = json.loads(condition_json)
    except Exception:
        return False, 0.0, {}

    cond = obj.get("cond", obj)  # bare condition allowed
    match = bool(_eval_node(cond, signals))

    def eval_expr(expr):
        return _value(expr, signals)

    severity = 0.5
    if "severity" in obj:
        try:
            s = eval_expr(obj["severity"])
            severity = float(s) if s is not None else 0.5
        except Exception:
            pass

    bindings: Dict[str, Any] = {}
    if "bindings" in obj:
        for k, expr in obj["bindings"].items():
            try:
                bindings[k] = eval_expr(expr)
            except Exception:
                bindings[k] = None

    return match, max(0.0, min(1.0, severity)), bindings


def choose_tone(tone_strategy_json: str, severity: float) -> str:
    try:
        obj = json.loads(tone_strategy_json)
    except Exception:
        return "gentle"

    for m in obj.get("map", []):
        if "gte" in m:
            key, thr = m["gte"]
            if key == "severity" and severity >= float(thr):
                return m.get("tone", "gentle")
        if "gt" in m:
            key, thr = m["gt"]
            if key == "severity" and severity > float(thr):
                return m.get("tone", "gentle")
    return obj.get("default", "gentle")


def derive_expectation(
    condition_json: str,
) -> tuple[list[str], Callable[[dict[str, float]], bool] | None]:

    """
    Derive (signal_names, expect_change_fn) from condition_json.
    Handles gte/gt/lt/lte/between, plus all/any.
    """
    
    try:
        obj = json.loads(condition_json)
    except Exception:
        return [], None

    cond = obj.get("cond", obj)

    # Extract signal checks
    def extract(cond) -> list[tuple[str, str, Any]]:
        checks: list[tuple[str, str, Any]] = []
        if isinstance(cond, dict):
            if "gte" in cond: k, thr = cond["gte"]; checks.append((k, "gte", float(thr)))
            elif "gt" in cond: k, thr = cond["gt"]; checks.append((k, "gt", float(thr)))
            elif "lte" in cond: k, thr = cond["lte"]; checks.append((k, "lte", float(thr)))
            elif "lt" in cond: k, thr = cond["lt"]; checks.append((k, "lt", float(thr)))
            elif "between" in cond:
                k, lo, hi = cond["between"]; checks.append((k, "between", (float(lo), float(hi))))
            elif "all" in cond:
                for c in cond["all"]: checks.extend(extract(c))
            elif "any" in cond:
                for c in cond["any"]: checks.extend(extract(c))
        return checks

    checks = extract(cond)
    signals = [c[0].split("signal:")[1] for c in checks if c[0].startswith("signal:")]
    
    def expect_change_fn(values: dict[str, float]) -> bool:
        results: list[bool] = []
        for k, op, thr in checks:
            if not k.startswith("signal:"):
                continue
            sig = k.split("signal:")[1]
            val = float(values.get(sig) or 0)

            if op in ("gte", "gt"):
                results.append(val < float(thr))
            elif op in ("lte", "lt"):
                results.append(val > float(thr))
            elif op == "between":
                lo, hi = thr
                results.append(val < lo or val > hi)

        return all(results) if isinstance(cond, dict) and "all" in cond else any(results)

    if checks:
        return signals, expect_change_fn
    else:
        return [], None