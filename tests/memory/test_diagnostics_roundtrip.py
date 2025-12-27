from datetime import datetime, UTC


def test_diagnostics_roundtrip(memstore):
    started = datetime.now(UTC).isoformat()

    memstore.add_diagnostic_event(
        mode="unit",
        fix=False,
        base="memory",
        diag_output="hello diagnostics",
        issues=[{"id": "X", "msg": "y"}],
        benchmarks=[],
        laggy=False,
        started_at_iso=started,
        duration_ms=12.3,
    )

    last = memstore.last_diagnostic()
    assert last is not None
    assert last.get("type") == "diagnostic"
    assert last.get("mode") == "unit"
    assert "hello diagnostics" in last.get("tool_output", "")
