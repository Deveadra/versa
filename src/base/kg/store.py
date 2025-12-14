from __future__ import annotations

from datetime import datetime

from ..database.sqlite import SQLiteConn
from .entities import DEFAULT_TYPE, ENTITY_TYPES
from .relations import RELATION_CONSTRAINTS, RELATION_INVERSES, RELATION_SYNONYMS

# A uniform row type for relations everywhere
RelationRow = tuple[str, str, str, float, str | None, str | None]


class KGStore:
    def __init__(self, db: SQLiteConn):
        self.db = db
        self._init_tables()

    def _init_tables(self) -> None:
        self.db.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                relation TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                valid_from TEXT DEFAULT CURRENT_TIMESTAMP,
                valid_to TEXT,
                FOREIGN KEY(source_id) REFERENCES entities(id),
                FOREIGN KEY(target_id) REFERENCES entities(id)
            );

            CREATE TABLE IF NOT EXISTS aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER NOT NULL,
                alias TEXT NOT NULL,
                FOREIGN KEY(entity_id) REFERENCES entities(id)
            );
            """
        )
        self.db.conn.commit()

    def upsert_entity(self, name: str, type_: str | None = None) -> int:
        # validate type
        if not type_ or type_ not in ENTITY_TYPES:
            type_ = DEFAULT_TYPE

        cur = self.db.conn.execute("SELECT id FROM entities WHERE name=?", (name,))
        row = cur.fetchone()
        if row:
            # row["id"] is Any; cast via int()
            return int(row["id"])

        cur = self.db.conn.execute("INSERT INTO entities(name, type) VALUES(?, ?)", (name, type_))
        self.db.conn.commit()
        return int(cur.lastrowid)  # type: ignore[attr-defined] # lastrowid is int|None in typeshed; after INSERT it's not None

    def add_relation(
        self,
        source_id: int,
        target_id: int,
        relation: str,
        confidence: float = 1.0,
        valid_from: str | None = None,
        valid_to: str | None = None,
    ) -> int:
        # load entity types
        row = self.db.conn.execute("SELECT type FROM entities WHERE id=?", (source_id,)).fetchone()
        if not row:
            return -1
        src_type = row["type"]
        row = self.db.conn.execute("SELECT type FROM entities WHERE id=?", (target_id,)).fetchone()
        if not row:
            return -1
        tgt_type = row["type"]

        canonical = RELATION_SYNONYMS.get(relation.lower(), relation.lower())

        # constraints
        if canonical in RELATION_CONSTRAINTS:
            expected_src, expected_tgt = RELATION_CONSTRAINTS[canonical]
            if src_type != expected_src or tgt_type != expected_tgt:
                print(f"[KG WARNING] Invalid relation: {src_type} -[{canonical}]-> {tgt_type}")
                return -1

        # close conflicting active relation
        row = self.db.conn.execute(
            """SELECT id, target_id FROM relations
            WHERE source_id=? AND relation=? AND valid_to IS NULL""",
            (source_id, canonical),
        ).fetchone()
        if row and row["target_id"] != target_id:
            self.db.conn.execute(
                "UPDATE relations SET valid_to=? WHERE id=?",
                (datetime.utcnow().isoformat(), row["id"]),
            )

        vfrom = valid_from or datetime.utcnow().isoformat()
        cur = self.db.conn.execute(
            """INSERT INTO relations(source_id, target_id, relation, confidence, valid_from, valid_to)
            VALUES(?, ?, ?, ?, ?, ?)""",
            (source_id, target_id, canonical, confidence, vfrom, valid_to),
        )
        rid = int(cur.lastrowid or -1)

        # inverse (best effort)
        if canonical in RELATION_INVERSES:
            inverse = RELATION_INVERSES[canonical]
            self.db.conn.execute(
                """INSERT INTO relations(source_id, target_id, relation, confidence, valid_from, valid_to)
                VALUES(?, ?, ?, ?, ?, ?)""",
                (target_id, source_id, inverse, confidence, vfrom, valid_to),
            )

        self.db.conn.commit()
        return rid

    def add_alias(self, entity_id: int, alias: str) -> None:
        self.db.conn.execute(
            "INSERT INTO aliases(entity_id, alias) VALUES(?, ?)",
            (entity_id, alias),
        )
        self.db.conn.commit()

    def list_aliases(self, entity_name: str) -> list[str]:
        entity_id = self.resolve_entity(entity_name)
        if not entity_id:
            return []
        cur = self.db.conn.execute("SELECT alias FROM aliases WHERE entity_id=?", (entity_id,))
        return [row["alias"] for row in cur.fetchall()]

    def resolve_entity(self, name: str) -> int | None:
        cur = self.db.conn.execute("SELECT id FROM entities WHERE name=?", (name,))
        row = cur.fetchone()
        if row:
            return int(row["id"])
        cur = self.db.conn.execute("SELECT entity_id FROM aliases WHERE alias=?", (name,))
        row = cur.fetchone()
        if row:
            return int(row["entity_id"])
        return None

    # ---------- queries (all return 6-tuples) ----------
    def query_relations(self, entity_name: str, at_time: str | None = None) -> list[RelationRow]:
        sql = """
        SELECT e1.name, r.relation, e2.name, r.confidence, r.valid_from, r.valid_to
        FROM relations r
        JOIN entities e1 ON r.source_id = e1.id
        JOIN entities e2 ON r.target_id = e2.id
        WHERE e1.name = ?
        """
        params: list[object] = [entity_name]
        if at_time:
            sql += " AND (r.valid_to IS NULL OR r.valid_to >= ?)"
            params.append(at_time)

        cur = self.db.conn.execute(sql, params)
        rows = cur.fetchall()
        return [(row[0], row[1], row[2], float(row[3]), row[4], row[5]) for row in rows]

    def query_relations_incoming(self, entity_name: str) -> list[RelationRow]:
        """Incoming edges to entity_name (same 6-field shape)."""
        sql = """
        SELECT e1.name, r.relation, e2.name, r.confidence, r.valid_from, r.valid_to
        FROM relations r
        JOIN entities e1 ON r.source_id = e1.id
        JOIN entities e2 ON r.target_id = e2.id
        WHERE e2.name = ?
        """
        cur = self.db.conn.execute(sql, (entity_name,))
        rows = cur.fetchall()
        return [(row[0], row[1], row[2], float(row[3]), row[4], row[5]) for row in rows]

    def query_past_relations(
        self, entity_name: str, relation_type: str | None = None
    ) -> list[RelationRow]:
        sql = """
        SELECT e1.name, r.relation, e2.name, r.confidence, r.valid_from, r.valid_to
        FROM relations r
        JOIN entities e1 ON r.source_id = e1.id
        JOIN entities e2 ON r.target_id = e2.id
        WHERE e1.name = ? AND r.valid_to IS NOT NULL
        """
        params: list[object] = [entity_name]
        if relation_type:
            sql += " AND r.relation = ?"
            params.append(relation_type)

        cur = self.db.conn.execute(sql, params)
        rows = cur.fetchall()
        return [(row[0], row[1], row[2], float(row[3]), row[4], row[5]) for row in rows]

    def query_future_relations(self, entity_name: str) -> list[RelationRow]:
        now = datetime.utcnow().isoformat()
        sql = """
        SELECT e1.name, r.relation, e2.name, r.confidence, r.valid_from, r.valid_to
        FROM relations r
        JOIN entities e1 ON r.source_id = e1.id
        JOIN entities e2 ON r.target_id = e2.id
        WHERE e1.name = ? AND r.valid_from > ?
        """
        cur = self.db.conn.execute(sql, (entity_name, now))
        rows = cur.fetchall()
        return [(row[0], row[1], row[2], float(row[3]), row[4], row[5]) for row in rows]

    # ---------- traversal ----------
    def _neighbors(self, entity_name: str, direction: str) -> list[RelationRow]:
        if direction == "out":
            return self.query_relations(entity_name)
        if direction == "in":
            return self.query_relations_incoming(entity_name)
        # both
        return self.query_relations(entity_name) + self.query_relations_incoming(entity_name)

    def multi_hop(
        self,
        start: str,
        max_hops: int = 3,
        direction: str = "both",
        at_time: str | None = None,
    ) -> list[list[RelationRow]]:
        paths: list[list[RelationRow]] = []
        visited: set[RelationRow] = set()

        def dfs(current: str, path: list[RelationRow], depth: int) -> None:
            if depth > max_hops:
                return
            neighbors = self._neighbors(current, direction)
            for src, rel, tgt, conf, vfrom, vto in neighbors:
                if at_time and vto and vto < at_time:
                    continue  # expired relation
                triple: RelationRow = (src, rel, tgt, conf, vfrom, vto)
                if triple in visited:
                    continue
                visited.add(triple)
                new_path = path + [triple]
                paths.append(new_path)
                next_node = tgt if src == current else src
                dfs(next_node, new_path, depth + 1)

        dfs(start, [], 1)
        return paths
