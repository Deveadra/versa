# tests/database/test_locking_and_retries.py

from __future__ import annotations

import sqlite3
import threading
import time


def test_busy_timeout_is_configured(db) -> None:
    ms = db.conn.execute("PRAGMA busy_timeout;").fetchone()[0]
    assert ms >= 1000, f"busy_timeout too low: {ms}ms (expected >= 1000ms)"


def test_concurrent_writer_waits_then_succeeds(tmp_path) -> None:
    """
    Writer A holds a write transaction. Writer B attempts to write.
    With busy_timeout set, B should wait and succeed once A commits.
    """
    db_path = tmp_path / "lock_wait.db"

    # Seed schema
    seed = sqlite3.connect(str(db_path), check_same_thread=False)
    seed.execute("PRAGMA journal_mode=WAL;")
    seed.execute("CREATE TABLE IF NOT EXISTS lock_test (id INTEGER)")
    seed.commit()
    seed.close()

    lock_acquired = threading.Event()
    release_lock = threading.Event()
    # result: dict[str, object] = {}
    result: dict[str, float | bool] = {}
    err: dict[str, Exception] = {}

    def locker() -> None:
        conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("BEGIN IMMEDIATE;")  # take write lock
            conn.execute("INSERT INTO lock_test(id) VALUES (1);")
            lock_acquired.set()
            release_lock.wait(timeout=5)
            conn.execute("COMMIT;")
        finally:
            conn.close()

    def writer() -> None:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        try:
            conn.execute("PRAGMA busy_timeout = 3000;")  # 3s
            start = time.time()
            conn.execute("INSERT INTO lock_test(id) VALUES (2);")
            conn.commit()
            result["ok"] = True
            result["elapsed"] = time.time() - start
        except Exception as e:
            result["ok"] = False
            err["e"] = e
            # result["err"] = e
        finally:
            conn.close()

    t1 = threading.Thread(target=locker, daemon=True)
    t1.start()
    assert lock_acquired.wait(timeout=2), "Locker thread did not acquire lock in time."

    t2 = threading.Thread(target=writer, daemon=True)
    t2.start()

    time.sleep(0.35)  # hold lock briefly
    release_lock.set()

    t2.join(timeout=5)
    t1.join(timeout=5)

    assert result.get("ok") is True, f"Writer failed unexpectedly: {err.get('e')}"
    assert float(result.get("elapsed", 0.0)) >= 0.1  # it waited at least a bit


def test_concurrent_writer_times_out(tmp_path) -> None:
    """
    Writer A holds the write lock longer than writer busy_timeout.
    Writer should fail with 'database is locked'.
    """
    db_path = tmp_path / "lock_timeout.db"

    seed = sqlite3.connect(str(db_path), check_same_thread=False)
    seed.execute("PRAGMA journal_mode=WAL;")
    seed.execute("CREATE TABLE IF NOT EXISTS lock_test (id INTEGER)")
    seed.commit()
    seed.close()

    lock_acquired = threading.Event()
    release_lock = threading.Event()
    err: dict[str, Exception] = {}

    def locker() -> None:
        conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute("INSERT INTO lock_test(id) VALUES (1);")
            lock_acquired.set()
            release_lock.wait(timeout=5)  # hold longer than writer timeout
            conn.execute("COMMIT;")
        finally:
            conn.close()

    def writer() -> None:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        try:
            conn.execute("PRAGMA busy_timeout = 100;")  # 0.1s
            conn.execute("INSERT INTO lock_test(id) VALUES (2);")
            conn.commit()
        except Exception as e:
            err["e"] = e
        finally:
            conn.close()

    t1 = threading.Thread(target=locker, daemon=True)
    t1.start()
    assert lock_acquired.wait(timeout=2), "Locker thread did not acquire lock in time."

    t2 = threading.Thread(target=writer, daemon=True)
    t2.start()
    t2.join(timeout=5)

    time.sleep(0.5)
    release_lock.set()
    t1.join(timeout=5)

    assert "e" in err, "Writer unexpectedly succeeded; expected a lock timeout."
    assert isinstance(err["e"], sqlite3.OperationalError)
    assert "locked" in str(err["e"]).lower()
