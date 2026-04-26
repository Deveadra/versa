from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

UTC = dt.UTC
DEFAULT_USER_AGENT = "AerithUltronClaudeWatch/0.1 (+local-monitor)"
KEYWORD_WEIGHTS = {
    "dream": 5,
    "kairos": 5,
    "memory": 4,
    "consolidate": 3,
    "rewrite": 3,
    "background agent": 4,
    "background": 2,
    "self-improve": 4,
    "self improve": 4,
    "agent": 2,
    "search": 2,
    "webfetch": 3,
    "web_search": 4,
    "tool": 1,
    "mcp": 4,
    "plugin": 3,
    "hooks": 3,
    "prompt": 2,
    "cache": 2,
    "cost": 2,
    "session": 2,
    "harness": 3,
    "permissions": 2,
    "oauth": 2,
    "sdk": 2,
    "voice": 2,
    "todo": 2,
    "security": 3,
    "malware": 5,
    "source map": 4,
    "leak": 2,
    "verified": 2,
}
HIGH_SIGNAL_PATHS = {
    "CHANGELOG.md": 7,
    "README.md": 4,
    "plugins/": 5,
    ".claude/": 4,
    "examples/": 3,
    "scripts/": 3,
    "sdk": 2,
    "mcp": 4,
    "search": 3,
    "auth": 3,
    "web": 2,
}
OFFICIAL_TAGS = {"official_release", "official_compare", "official_commit"}
COMMUNITY_TAGS = {"community_claim", "security_warning", "implementation_observation"}


@dataclasses.dataclass(slots=True)
class Config:
    github_owner: str = os.getenv("CLAUDE_WATCH_GITHUB_OWNER", "anthropics")
    github_repo: str = os.getenv("CLAUDE_WATCH_GITHUB_REPO", "claude-code")
    github_branch: str = os.getenv("CLAUDE_WATCH_GITHUB_BRANCH", "main")
    github_token: str | None = os.getenv("GITHUB_TOKEN")
    reddit_thread_url: str = os.getenv(
        "CLAUDE_WATCH_REDDIT_THREAD_URL",
        "https://www.reddit.com/r/ClaudeAI/comments/1s9d9j9/claude_code_source_leak_megathread/",
    )
    state_db_path: Path = Path(os.getenv("CLAUDE_WATCH_STATE_DB", "./state/claude_watch.db"))
    output_dir: Path = Path(os.getenv("CLAUDE_WATCH_OUTPUT_DIR", "./reports/claude_watch"))
    poll_seconds: int = int(os.getenv("CLAUDE_WATCH_POLL_SECONDS", "1800"))
    user_agent: str = os.getenv("CLAUDE_WATCH_USER_AGENT", DEFAULT_USER_AGENT)
    request_timeout: int = int(os.getenv("CLAUDE_WATCH_TIMEOUT_SECONDS", "30"))
    llm_analysis_command: str | None = os.getenv("CLAUDE_WATCH_ANALYSIS_CMD")
    max_release_fetch: int = int(os.getenv("CLAUDE_WATCH_MAX_RELEASES", "6"))
    max_commit_fetch: int = int(os.getenv("CLAUDE_WATCH_MAX_COMMITS", "30"))
    max_reddit_comments_per_run: int = int(os.getenv("CLAUDE_WATCH_MAX_REDDIT_COMMENTS", "1000"))


class HttpClient:
    def __init__(self, config: Config):
        self.config = config

    def get_json(self, url: str, *, headers: dict[str, str] | None = None) -> Any:
        req = urllib.request.Request(url, headers=self._headers(headers))
        return self._request_json(req)

    def _request_json(self, req: urllib.request.Request) -> Any:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.config.request_timeout) as response:
                    payload = response.read().decode("utf-8")
                    return json.loads(payload)
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                if isinstance(exc, urllib.error.HTTPError) and exc.code in {401, 403, 404}:
                    break
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"request failed for {req.full_url}: {last_error}")

    def _headers(self, headers: dict[str, str] | None) -> dict[str, str]:
        merged = {
            "Accept": "application/vnd.github+json, application/json;q=0.9",
            "User-Agent": self.config.user_agent,
        }
        if self.config.github_token:
            merged["Authorization"] = f"Bearer {self.config.github_token}"
        if headers:
            merged.update(headers)
        return merged


class StateStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cursors (
                source TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS seen_items (
                source TEXT NOT NULL,
                item_id TEXT NOT NULL,
                seen_at TEXT NOT NULL,
                PRIMARY KEY (source, item_id)
            );

            CREATE TABLE IF NOT EXISTS findings (
                finding_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                item_id TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence TEXT NOT NULL,
                significance INTEGER NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                raw_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def get_cursor(self, source: str) -> str | None:
        row = self.conn.execute("SELECT value FROM cursors WHERE source = ?", (source,)).fetchone()
        return row[0] if row else None

    def set_cursor(self, source: str, value: str) -> None:
        now = now_iso()
        self.conn.execute(
            """
            INSERT INTO cursors(source, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(source) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (source, value, now),
        )
        self.conn.commit()

    def has_seen(self, source: str, item_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM seen_items WHERE source = ? AND item_id = ?",
            (source, item_id),
        ).fetchone()
        return row is not None

    def mark_seen(self, source: str, item_id: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO seen_items(source, item_id, seen_at) VALUES (?, ?, ?)",
            (source, item_id, now_iso()),
        )

    def store_finding(self, finding: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO findings(
                finding_id, source, item_id, category, confidence, significance,
                title, summary, raw_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                finding["finding_id"],
                finding["source"],
                finding["item_id"],
                finding["category"],
                finding["confidence"],
                int(finding["significance"]),
                finding["title"],
                finding["summary"],
                json.dumps(finding, ensure_ascii=False),
                now_iso(),
            ),
        )

    def commit(self) -> None:
        self.conn.commit()


class GitHubCollector:
    def __init__(self, config: Config, http: HttpClient, state: StateStore):
        self.config = config
        self.http = http
        self.state = state

    @property
    def api_base(self) -> str:
        return f"https://api.github.com/repos/{self.config.github_owner}/{self.config.github_repo}"

    def collect(self) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        findings.extend(self._collect_releases())
        findings.extend(self._collect_repo_changes())
        self.state.commit()
        return findings

    def _collect_releases(self) -> list[dict[str, Any]]:
        url = f"{self.api_base}/releases?per_page={self.config.max_release_fetch}"
        releases = self.http.get_json(url)
        last_tag = self.state.get_cursor("github_last_release_tag")
        findings: list[dict[str, Any]] = []
        newest_tag: str | None = None
        for release in releases:
            tag = release.get("tag_name") or ""
            if newest_tag is None and tag:
                newest_tag = tag
            if last_tag and tag == last_tag:
                break
            item = {
                "source": "github",
                "source_kind": "release",
                "item_id": f"release:{tag}",
                "title": f"Release {tag}",
                "body": release.get("body") or "",
                "metadata": {
                    "html_url": release.get("html_url"),
                    "published_at": release.get("published_at"),
                    "author": (release.get("author") or {}).get("login"),
                    "prerelease": bool(release.get("prerelease")),
                    "draft": bool(release.get("draft")),
                },
            }
            findings.append(Analyzer.classify(item))
            self.state.mark_seen("github_release", tag)
        if newest_tag:
            self.state.set_cursor("github_last_release_tag", newest_tag)
        return findings

    def _collect_repo_changes(self) -> list[dict[str, Any]]:
        branch = urllib.parse.quote(self.config.github_branch)
        head = self.http.get_json(f"{self.api_base}/commits/{branch}")
        head_sha = head["sha"]
        last_sha = self.state.get_cursor("github_last_commit_sha")
        findings: list[dict[str, Any]] = []

        if not last_sha:
            recent = self.http.get_json(
                f"{self.api_base}/commits?sha={branch}&per_page={self.config.max_commit_fetch}"
            )
            grouped = {
                "source": "github",
                "source_kind": "baseline_commits",
                "item_id": f"baseline:{head_sha}",
                "title": f"Baseline commit snapshot at {head_sha[:7]}",
                "body": "\n".join(
                    f"- {commit['sha'][:7]} {commit['commit']['message'].splitlines()[0]}"
                    for commit in recent[: min(10, len(recent))]
                ),
                "metadata": {
                    "head_sha": head_sha,
                    "commit_count": len(recent),
                    "commit_messages": [commit["commit"]["message"] for commit in recent],
                    "files": [],
                },
            }
            findings.append(Analyzer.classify(grouped))
            self.state.mark_seen("github_compare", grouped["item_id"])
            self.state.set_cursor("github_last_commit_sha", head_sha)
            return findings

        if last_sha == head_sha:
            return findings

        compare_url = f"{self.api_base}/compare/{last_sha}...{head_sha}"
        compare = self.http.get_json(compare_url)
        files = [f["filename"] for f in compare.get("files", [])]
        commits = compare.get("commits", [])
        item = {
            "source": "github",
            "source_kind": "compare",
            "item_id": f"compare:{last_sha[:7]}..{head_sha[:7]}",
            "title": f"Repo advanced from {last_sha[:7]} to {head_sha[:7]}",
            "body": "\n".join(
                f"- {commit['sha'][:7]} {commit['commit']['message'].splitlines()[0]}"
                for commit in commits
            ),
            "metadata": {
                "from_sha": last_sha,
                "to_sha": head_sha,
                "html_url": compare.get("html_url"),
                "ahead_by": compare.get("ahead_by", 0),
                "total_commits": compare.get("total_commits", len(commits)),
                "files": files,
            },
        }
        findings.append(Analyzer.classify(item))
        for commit in commits:
            commit_item = {
                "source": "github",
                "source_kind": "commit",
                "item_id": f"commit:{commit['sha']}",
                "title": f"Commit {commit['sha'][:7]}",
                "body": commit["commit"]["message"],
                "metadata": {
                    "html_url": commit.get("html_url"),
                    "author": (commit.get("author") or {}).get("login"),
                    "timestamp": commit["commit"]["author"].get("date"),
                    "files": files,
                },
            }
            findings.append(Analyzer.classify(commit_item))
            self.state.mark_seen("github_commit", commit["sha"])
        self.state.mark_seen("github_compare", item["item_id"])
        self.state.set_cursor("github_last_commit_sha", head_sha)
        return findings


class RedditCollector:
    def __init__(self, config: Config, http: HttpClient, state: StateStore):
        self.config = config
        self.http = http
        self.state = state

    def collect(self) -> list[dict[str, Any]]:
        url = self._thread_json_url(self.config.reddit_thread_url)
        payload = self.http.get_json(url, headers={"Accept": "application/json"})
        if not isinstance(payload, list) or len(payload) < 2:
            raise RuntimeError("unexpected reddit payload shape")

        post_listing = payload[0]
        comments_listing = payload[1]
        post_data = post_listing["data"]["children"][0]["data"]
        post_id = post_data.get("id")
        post_title = post_data.get("title") or "Reddit thread"

        comments = flatten_reddit_comments(comments_listing["data"]["children"])
        comments = comments[: self.config.max_reddit_comments_per_run]
        findings: list[dict[str, Any]] = []

        for comment in comments:
            comment_id = comment.get("id")
            if not comment_id or self.state.has_seen("reddit_comment", comment_id):
                continue
            body = comment.get("body") or ""
            author = comment.get("author") or "[deleted]"
            title = f"Reddit comment by {author}"
            item = {
                "source": "reddit",
                "source_kind": "comment",
                "item_id": f"comment:{comment_id}",
                "title": title,
                "body": body,
                "metadata": {
                    "author": author,
                    "score": int(comment.get("score") or 0),
                    "replies_count": count_replies(comment),
                    "created_utc": comment.get("created_utc"),
                    "permalink": f"https://www.reddit.com{comment.get('permalink', '')}",
                    "thread_post_id": post_id,
                    "thread_title": post_title,
                    "distinguished": comment.get("distinguished"),
                },
            }
            finding = Analyzer.classify(item)
            if finding["significance"] >= 5 or finding["category"] in {
                "security_warning",
                "implementation_observation",
            }:
                findings.append(finding)
            self.state.mark_seen("reddit_comment", comment_id)
        self.state.set_cursor("reddit_last_scan_utc", now_iso())
        self.state.commit()
        return findings

    @staticmethod
    def _thread_json_url(thread_url: str) -> str:
        clean = thread_url.rstrip("/")
        if clean.endswith(".json"):
            return clean + "?sort=new&limit=500"
        return clean + ".json?sort=new&limit=500"


class Analyzer:
    @staticmethod
    def classify(item: dict[str, Any]) -> dict[str, Any]:
        body = normalize_whitespace(item.get("body") or "")
        title = normalize_whitespace(item.get("title") or "")
        haystack = f"{title}\n{body}".lower()
        metadata = item.get("metadata", {})
        source = item["source"]
        source_kind = item.get("source_kind", "unknown")

        significance = 0
        rationale: list[str] = []
        takeaways: list[str] = []

        for keyword, weight in KEYWORD_WEIGHTS.items():
            if keyword in haystack:
                significance += weight
                rationale.append(f"keyword:{keyword}")

        if source == "github":
            significance += 5
            if source_kind == "release":
                significance += 10
                category = "official_release"
                confidence = "high"
                takeaways.append(
                    "Treat release notes as authoritative signal for productized features and fixes."
                )
            elif source_kind == "compare":
                significance += 8
                category = "official_compare"
                confidence = "high"
            else:
                significance += 4
                category = "official_commit"
                confidence = "high"

            files = metadata.get("files") or []
            path_hits = score_paths(files)
            significance += path_hits[0]
            rationale.extend(path_hits[1])
            if path_hits[1]:
                takeaways.append(
                    "Changed files touch high-signal product surfaces worth diffing in Aerith's research pipeline."
                )

        else:
            score = int(metadata.get("score") or 0)
            replies = int(metadata.get("replies_count") or 0)
            if score >= 25:
                significance += 4
                rationale.append("community_score>=25")
            elif score >= 5:
                significance += 2
                rationale.append("community_score>=5")
            if replies >= 5:
                significance += 2
                rationale.append("replies>=5")
            if "http://" in body or "https://" in body:
                significance += 2
                rationale.append("links_present")
            if code_identifier_count(body) >= 2:
                significance += 3
                rationale.append("contains_code_identifiers")

            category = "community_claim"
            confidence = "low"
            if any(
                token in haystack
                for token in ["verified", "source is", "i checked", "minified", "npm source"]
            ):
                confidence = "medium"
            if any(
                token in haystack
                for token in [
                    "malware",
                    "compromise",
                    "attack window",
                    "credential",
                    "rotate all credentials",
                ]
            ):
                category = "security_warning"
                confidence = "medium"
                significance += 5
                takeaways.append(
                    "Security note: treat this as operational caution, not product inspiration."
                )
            elif any(
                token in haystack
                for token in ["hardcoded", "converter", "pagination", "quote maximum", "body only"]
            ):
                category = "implementation_observation"
                confidence = "medium"
                significance += 3
                takeaways.append(
                    "Potentially useful implementation detail; queue for source verification before adopting."
                )
            elif any(
                token in haystack
                for token in [
                    "dream mode",
                    "kairos_dream",
                    "background agent",
                    "rewrites its own memory",
                ]
            ):
                category = "community_claim"
                confidence = "medium"
                significance += 5
                takeaways.append(
                    "Strong concept match for Aerith's dream/consolidation loop, but keep this quarantined as unverified until corroborated elsewhere."
                )

        significance = min(significance, 100)
        summary = summarize_item(source, category, title, body, metadata)
        llm_augmented = maybe_run_external_analyzer(item, significance, category, confidence)
        if llm_augmented:
            significance = int(llm_augmented.get("significance", significance))
            category = llm_augmented.get("category", category)
            confidence = llm_augmented.get("confidence", confidence)
            summary = llm_augmented.get("summary", summary)
            takeaways = llm_augmented.get("takeaways", takeaways) or takeaways
            rationale = llm_augmented.get("rationale", rationale) or rationale

        finding_id = sha256_json(
            {
                "source": source,
                "item_id": item["item_id"],
                "category": category,
                "summary": summary,
            }
        )
        return {
            "finding_id": finding_id,
            "source": source,
            "source_kind": source_kind,
            "item_id": item["item_id"],
            "category": category,
            "confidence": confidence,
            "significance": significance,
            "title": title,
            "summary": summary,
            "rationale": rationale,
            "takeaways": takeaways,
            "metadata": metadata,
            "body_excerpt": clip(body, 500),
        }


def maybe_run_external_analyzer(
    item: dict[str, Any],
    significance: int,
    category: str,
    confidence: str,
) -> dict[str, Any] | None:
    cmd = os.getenv("CLAUDE_WATCH_ANALYSIS_CMD")
    if not cmd:
        return None
    envelope = {
        "item": item,
        "heuristic": {
            "significance": significance,
            "category": category,
            "confidence": confidence,
        },
    }
    try:
        completed = subprocess.run(
            cmd,
            input=json.dumps(envelope, ensure_ascii=False),
            capture_output=True,
            text=True,
            shell=True,
            check=True,
        )
        stdout = completed.stdout.strip()
        return json.loads(stdout) if stdout else None
    except Exception:
        return None


class DigestWriter:
    def __init__(self, config: Config, state: StateStore):
        self.config = config
        self.state = state
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, findings: list[dict[str, Any]]) -> tuple[Path, Path]:
        ts = dt.datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        json_path = self.config.output_dir / f"digest_{ts}.json"
        md_path = self.config.output_dir / f"digest_{ts}.md"

        findings = sorted(findings, key=lambda x: x["significance"], reverse=True)
        for finding in findings:
            self.state.store_finding(finding)
        self.state.commit()

        json_path.write_text(json.dumps(findings, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(render_markdown(findings), encoding="utf-8")
        return md_path, json_path


class ClaudeWatchAgent:
    def __init__(self, config: Config):
        self.config = config
        self.state = StateStore(config.state_db_path)
        self.http = HttpClient(config)
        self.github = GitHubCollector(config, self.http, self.state)
        self.reddit = RedditCollector(config, self.http, self.state)
        self.writer = DigestWriter(config, self.state)

    def close(self) -> None:
        self.state.close()

    def run_once(self) -> dict[str, Any]:
        findings: list[dict[str, Any]] = []
        errors: list[str] = []

        for name, collector in (("github", self.github.collect), ("reddit", self.reddit.collect)):
            try:
                findings.extend(collector())
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        md_path, json_path = self.writer.write(findings)
        return {
            "timestamp": now_iso(),
            "finding_count": len(findings),
            "markdown_digest": str(md_path),
            "json_digest": str(json_path),
            "errors": errors,
        }

    def run_forever(self) -> None:
        while True:
            result = self.run_once()
            print(json.dumps(result, indent=2))
            time.sleep(self.config.poll_seconds)


def render_markdown(findings: list[dict[str, Any]]) -> str:
    official = [f for f in findings if f["category"] in OFFICIAL_TAGS]
    community = [
        f for f in findings if f["category"] in COMMUNITY_TAGS or f["category"] == "community_claim"
    ]
    important = [f for f in findings if f["significance"] >= 12]

    lines = [
        "# Claude Watch Digest",
        "",
        f"Generated: {now_iso()}",
        "",
        "> Official GitHub signals are authoritative. Reddit findings are treated as community observations unless separately verified.",
        "",
        f"- Total findings: {len(findings)}",
        f"- High-significance findings: {len(important)}",
        f"- Official findings: {len(official)}",
        f"- Community findings: {len(community)}",
        "",
    ]

    if official:
        lines.extend(["## Official repo activity", ""])
        for finding in official[:20]:
            lines.extend(render_finding_block(finding))

    if community:
        lines.extend(["## Community observations and claims", ""])
        for finding in community[:30]:
            lines.extend(render_finding_block(finding))

    if important:
        lines.extend(["## Aerith/Ultron queue candidates", ""])
        for finding in important[:15]:
            lines.append(f"- **{finding['title']}** — {finding['summary']}")
        lines.append("")

    if not findings:
        lines.extend(
            [
                "## No new findings",
                "",
                "No new releases, repo deltas, or significant Reddit comments were captured this run.",
                "",
            ]
        )

    return "\n".join(lines)


def render_finding_block(finding: dict[str, Any]) -> list[str]:
    lines = [
        f"### {finding['title']}",
        "",
        f"- Category: `{finding['category']}`",
        f"- Confidence: `{finding['confidence']}`",
        f"- Significance: `{finding['significance']}`",
    ]
    metadata = finding.get("metadata", {})
    url = metadata.get("html_url") or metadata.get("permalink")
    if url:
        lines.append(f"- Link: {url}")
    if finding.get("takeaways"):
        lines.append(f"- Takeaway: {finding['takeaways'][0]}")
    lines.append("")
    lines.append(finding["summary"])
    lines.append("")
    excerpt = finding.get("body_excerpt")
    if excerpt:
        lines.append("> " + excerpt.replace("\n", "\n> "))
        lines.append("")
    return lines


def summarize_item(
    source: str,
    category: str,
    title: str,
    body: str,
    metadata: dict[str, Any],
) -> str:
    sentence = clip(body.splitlines()[0] if body else title, 240)
    if source == "github" and category == "official_release":
        published = metadata.get("published_at") or "unknown time"
        return f"New official release detected. Published at {published}. Primary note: {sentence}"
    if source == "github" and category == "official_compare":
        files = metadata.get("files") or []
        hot = ", ".join(files[:6]) if files else "no changed files listed"
        return f"Official repo advanced with {metadata.get('total_commits', 0)} commits. Notable changed paths: {hot}."
    if source == "reddit" and category == "security_warning":
        return f"Community security warning surfaced in the megathread. Treat as caution until verified independently. First note: {sentence}"
    if source == "reddit":
        return f"Community observation from the megathread. Preserve as a lead, not ground truth. First note: {sentence}"
    return sentence


def score_paths(files: Iterable[str]) -> tuple[int, list[str]]:
    score = 0
    hits: list[str] = []
    for file_path in files:
        for marker, weight in HIGH_SIGNAL_PATHS.items():
            if marker in file_path:
                score += weight
                hits.append(f"path:{marker}")
    return score, hits


def flatten_reddit_comments(children: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comments: list[dict[str, Any]] = []
    stack = list(children)
    while stack:
        node = stack.pop(0)
        if node.get("kind") != "t1":
            continue
        data = node.get("data") or {}
        comments.append(data)
        replies = data.get("replies")
        if isinstance(replies, dict):
            more = replies.get("data", {}).get("children", [])
            stack[0:0] = more
    return comments


def count_replies(comment: dict[str, Any]) -> int:
    replies = comment.get("replies")
    if not isinstance(replies, dict):
        return 0
    return len(replies.get("data", {}).get("children", []) or [])


def code_identifier_count(text: str) -> int:
    return len(re.findall(r"[A-Z_]{3,}|`[^`]+`", text))


def clip(text: str, limit: int) -> str:
    text = normalize_whitespace(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def now_iso() -> str:
    return dt.datetime.now(UTC).replace(microsecond=0).isoformat()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor Claude Code repo activity and a Reddit megathread."
    )
    parser.add_argument("--once", action="store_true", help="Run a single polling cycle and exit.")
    parser.add_argument(
        "--loop", action="store_true", help="Run forever on the configured poll interval."
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the active configuration and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    config = Config()
    if args.print_config:
        payload = dataclasses.asdict(config)
        payload["state_db_path"] = str(config.state_db_path)
        payload["output_dir"] = str(config.output_dir)
        print(json.dumps(payload, indent=2))
        return 0

    agent = ClaudeWatchAgent(config)
    try:
        if args.loop:
            agent.run_forever()
            return 0
        result = agent.run_once()
        print(json.dumps(result, indent=2))
        return 0 if not result["errors"] else 1
    finally:
        agent.close()


if __name__ == "__main__":
    raise SystemExit(main())
