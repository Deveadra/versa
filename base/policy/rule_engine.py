from __future__ import annotations
import json, re
from typing import Dict, Any, Tuple

def _coerce(val: str) -> Any:
    s = str(val).strip().lower()
    if s in ("true","false"): return s == "true"
    try:
        if "." in s: return float(s)
        return int(s)
    except Exception:
        return val

def _get_signal(signals: Dict[str, dict], key: str) -> Any:
    # key may be "signal:name" or "name"
    name = key.split("signal:",1)[-1]
    entry = signals.get(name)
    if not entry: return None
    return _coerce(entry.get("value"))

def _eval_node(node: Any, signals: Dict[str,dict]) -> Any:
    if isinstance(node, (str,int,float,bool)) or node is None:
        return node
    if isinstance(node, list):
        return [_eval_node(n, signals) for n in node]
    if isinstance(node, dict):
        # JSON-logic like ops
        if "all" in node:
            return all(_eval_node(x, signals) for x in node["all"])
        if "any" in node:
            return any(_eval_node(x, signals) for x in node["any"])
        if "not" in node:
            return not _eval_node(node["not"], signals)
        if "eq" in node:
            a, b = node["eq"]
            a = _value(a, signals); b = _value(b, signals)
            return a == b
        if "neq" in node:
            a, b = node["neq"]; a = _value(a, signals); b = _value(b, signals)
            return a != b
        if "gt" in node:
            a, b = node["gt"]; a = float(_value(a, signals) or 0); b = float(_value(b, signals) or 0)
            return a > b
        if "gte" in node:
            a, b = node["gte"]; a = float(_value(a, signals) or 0); b = float(_value(b, signals) or 0)
            return a >= b
        if "lt" in node:
            a, b = node["lt"]; a = float(_value(a, signals) or 0); b = float(_value(b, signals) or 0)
            return a < b
        if "lte" in node:
            a, b = node["lte"]; a = float(_value(a, signals) or 0); b = float(_value(b, signals) or 0)
            return a <= b
        if "exists" in node:
            k = node["exists"]
            return _get_signal(signals, k) is not None
        if "regex" in node:
            val, pattern = node["regex"]
            v = str(_value(val, signals) or "")
            return re.search(pattern, v) is not None
        if "between" in node:
            x, lo, hi = node["between"]
            v = float(_value(x, signals) or 0)
            return (float(_value(lo, signals) or 0) <= v <= float(_value(hi, signals) or 0))
        # passthrough
        return {k:_eval_node(v, signals) for k,v in node.items()}
    return None

def _value(node: Any, signals: Dict[str,dict]) -> Any:
    if isinstance(node, str) and node.startswith("signal:"):
        return _get_signal(signals, node)
    return _eval_node(node, signals)

def evaluate_condition(condition_json: str, signals: Dict[str,dict]) -> Tuple[bool, float, Dict[str,Any]]:
    """
    Returns (match, severity, bindings)
    - severity is optional; if not present in condition JSON, defaults to 0.5
    - bindings can include any intermediate computed fields for templating
    """
    try:
        obj = json.loads(condition_json)
    except Exception:
        return False, 0.0, {}

    # convention: {"cond": {...}, "severity": <expr>, "bindings": {"x": <expr>, ...}}
    cond = obj.get("cond", obj)  # allow bare condition or wrapped
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

    bindings = {}
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
    # simple map: first matching rule wins
    for m in obj.get("map", []):
        if "gte" in m:
            key, thr = m["gte"]
            if key == "severity" and severity >= float(thr): return m.get("tone","gentle")
        if "gt" in m:
            key, thr = m["gt"]
            if key == "severity" and severity > float(thr): return m.get("tone","gentle")
    return obj.get("default","gentle")


def derive_expectation(condition_json: str):
    """
    Derive (signal_names, expect_change_fn) from a rule's condition_json.
    Handles gte/gt/lt/lte/between, plus all/any.
    """
    import json
    try:
        obj = json.loads(condition_json)
    except Exception:
        return [], None

    cond = obj.get("cond", obj)  # unwrap

    # Collect all signal checks
    def extract(cond) -> list[tuple[str, str, float]]:
        checks = []
        if isinstance(cond, dict):
            if "gte" in cond:
                k, thr = cond["gte"]; checks.append((k, "gte", float(thr)))
            elif "gt" in cond:
                k, thr = cond["gt"]; checks.append((k, "gt", float(thr)))
            elif "lte" in cond:
                k, thr = cond["lte"]; checks.append((k, "lte", float(thr)))
            elif "lt" in cond:
                k, thr = cond["lt"]; checks.append((k, "lt", float(thr)))
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
        """values is {signal_name: current_value}"""
        ok = []
        for k, op, thr in checks:
            if not k.startswith("signal:"):
                continue
            sig = k.split("signal:")[1]
            val = float(values.get(sig) or 0)

            if op in ("gte", "gt"):
                # Expect improvement = drop below threshold
                ok.append(val < thr)
            elif op in ("lte", "lt"):
                # Expect improvement = rise above threshold
                ok.append(val > thr)
            elif op == "between":
                lo, hi = thr
                # Expect improvement = leave that risky range
                ok.append(val < lo or val > hi)
        return all(ok) if "all" in cond else any(ok)

    return signals, expect_change_fn if checks else ([], None)

