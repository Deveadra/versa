# tests/database/schema_contract.py


# tests/database/schema_contract.py
from __future__ import annotations

REQUIRED_TABLES: dict[str, set[str]] = {
    "schema_migrations": {"id", "filename", "applied_at"},
    "facts": {"id", "key", "value", "last_updated", "confidence", "last_reinforced", "embedding"},
    "events": {"id", "content", "ts", "importance", "type"},
    "usage_log": {
        "id",
        "user_text",
        "normalized_intent",
        "resolved_action",
        "params_json",
        "success",
        "latency_ms",
        "created_at",
    },
    "habits": {"id", "key", "count", "score", "last_used"},
    "feedback_events": {"id", "usage_id", "kind", "note", "created_at"},
    "policy_assignments": {"usage_id", "policy_id", "created_at"},
    "context_signals": {
        "id",
        "name",
        "value",
        "type",
        "description",
        "confidence",
        "source",
        "last_updated",
    },
    "derived_signals": {"id", "name", "definition", "created_at"},
    "engagement_rules": {
        "id",
        "name",
        "enabled",
        "topic_id",
        "priority",
        "reset_signals",
        "cooldown_seconds",
        "max_per_day",
        "condition_json",
        "tone_strategy_json",
        "context_template",
        "created_at",
        "updated_at",
    },
    "rule_stats": {"rule_id", "last_fired", "fires_today", "ema_success", "ema_negative"},
    "rule_history": {"id", "rule_id", "fired_at", "topic_id", "tone", "context", "outcome"},
    "rule_audit": {"id", "rule_name", "topic_id", "rationale", "details_json", "created_at"},
    "tone_memory": {
        "id",
        "topic_id",
        "tone",
        "ignored_count",
        "acted_count",
        "consequence_note",
        "last_updated",
        "last_tone",
        "last_outcome",
        "updated_at",
    },
    "topics": {"topic_id", "policy", "conviction", "created_at"},
    "topic_overrides": {"id", "topic_id", "type", "reason", "expires_at", "created_at"},
    "topic_state": {"topic_id", "ignore_count", "escalation_count", "last_mentioned"},
    "topic_feedback": {"id", "topic_id", "feedback", "created_at"},
    "consequence_map": {"id", "keyword", "topic_id", "confidence", "last_updated"},
    "complaint_clusters": {"id", "cluster", "topic_id", "examples", "last_updated", "last_example"},
    "proposed_rules": {
        "id",
        "name",
        "topic_id",
        "priority",
        "cooldown_seconds",
        "max_per_day",
        "condition_json",
        "tone_strategy_json",
        "context_template",
        "confidence",
        "score",
        "status",
        "rationale",
        "created_at",
        "approved_at",
        "denied_at",
        "applied_at",
        "reverted_at",
    },
}

# Virtual table: validate existence + basic query separately
REQUIRED_VIRTUAL_TABLES: set[str] = {"events_fts"}

# Migrations you expect present in repo
REQUIRED_MIGRATIONS: set[str] = {"0001_init.sql"}

REQUIRED_TABLES: dict[str, set[str]] = {
    # Fill these in once we confirm the real schema from 0001_init.sql
    # "events": {"id", "ts", "type", "payload"},
    # "usage_log": {"id", "ts", "action", "success"},
}

# Optional: indexes you consider critical for performance
# REQUIRED_INDEXES: set[str] = {
#     # e.g. "idx_usage_log_ts",
# }

# Optional: foreign key expectations (table -> referenced table)
REQUIRED_FOREIGN_KEYS: dict[str, set[str]] = {
    # "child_table": {"parent_table"},
}
