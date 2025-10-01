
1. Extend your nightly job to adjust rules and invent new ones from observed outcomes:
Outcomes (“acted/thanks/ignore/angry”) get written by you wherever you capture user behavior, via:

```bash
policy.conn.execute("INSERT INTO rule_history(rule_id, topic_id, tone, context, outcome) VALUES(?,?,?,?,?)",
                    (rule_id, topic, tone, context, outcome))
# Update EMAs in rule_stats accordingly (your feedback hook)
```

2. In Dream Cycle Code

When creating or refining rules:
```bash
cur.execute("INSERT OR IGNORE INTO topics (id) VALUES (?)", (topic_id,))
```

So if Ultron spawns a new rule about "screen_time", that topic is automatically registered.

3. Hook Mood to Tone Memory

When engagement manager prepares context:

```bash
tone = choose_tone_for_topic(self.db, row["topic_id"])
last_example = style_complaint(cluster_row["last_example"], mood=tone) if cluster_row else None
```

4. Review Options

Voice: “Ultron, approve rule 2.” → status='approved'.

Text: “Show me pending rules.” → he reads from DB.

Logs: Check logs/morning_review_YYYYMMDD.txt.

CLI (optional):

python manage.py review --list
python manage.py review --approve 3
python manage.py review --deny 4

5. 