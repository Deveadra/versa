import subprocess
import sys
from pathlib import Path

from base.llm.prompt_composer import (
    compose_persona_block,
    compose_prompt,
    compose_retrieval_block,
    justify_memory,
)

root = Path(__file__).parent.parent
tests_dir = root / "tests"
tests_dir.mkdir(exist_ok=True)
import unittest


# --- Dummy mocks for required deps ---
class Dummy:
    def __getattr__(self, item):
        return lambda *a, **k: None


dummy_profile = Dummy()
dummy_memory = Dummy()
dummy_habit = Dummy()


class PromptComposerTests(unittest.TestCase):
    def test_compose_persona_block_empty(self):
        self.assertEqual(compose_persona_block(""), "")
        self.assertEqual(compose_persona_block(None), "")

    def test_compose_persona_block_basic(self):
        persona = "Name: Sundance\npreferred_player: Spotify\nfavorite_music: lo-fi\n"
        block = compose_persona_block(persona)
        self.assertIn("Persona:", block)
        self.assertIn("Sundance", block)
        self.assertIn("Spotify", block)

    def test_justify_memory_and_retrieval_block(self):
        mems = [
            {
                "summary": "Sundance likes lo-fi on Spotify",
                "source": "usage_log",
                "score": 0.92,
                "last_used": "2025-09-10T12:00:00",
            },
            {
                "summary": "Prefers short replies in mornings",
                "source": "usage_log",
                "score": 0.6,
                "last_used": "2025-09-18T08:00:00",
            },
            {"summary": "Slept at 23:30 this week", "source": "facts", "score": 0.3},
        ]
        block = compose_retrieval_block(mems, top_k=2)
        self.assertIn("Sundance likes lo-fi", block)
        self.assertIn("Prefers short replies", block)
        justification = justify_memory(mems[0])
        self.assertIn("from usage_log", justification)
        self.assertIn("score=", justification)

    def test_compose_prompt_full(self):
        sys_text = "You are Ultron. Be concise and warm."
        persona = "Name: Sundance\npreferred_player: Spotify\nfavorite_music: lo-fi\n"
        mems = [
            {
                "summary": "Sundance likes lo-fi on Spotify",
                "source": "usage_log",
                "score": 0.92,
                "last_used": "2025-09-10T12:00:00",
            }
        ]
        prompt = compose_prompt(
            sys_text,
            "Play my usual",
            persona_text=persona,
            memories=mems,
            extra_context="KG: none",
            top_k_memories=1,
            profile_mgr=dummy_profile,  # type: ignore
            memory_store=dummy_memory,  # type: ignore
            habit_miner=dummy_habit,  # type: ignore
        )
        self.assertIn("SYSTEM:", prompt)
        self.assertIn("Persona:", prompt)
        self.assertIn("Relevant memories:", prompt)
        self.assertIn("User:", prompt)


if __name__ == "__main__":
    unittest.main()

# Save this test file's code to disk for discovery
test_code = Path(__file__).read_text(encoding="utf-8")
(test_path := tests_dir / "test_prompt_composer.py").write_text(test_code, encoding="utf-8")
print("Wrote test:", test_path)

# run tests
print("Running unit tests with unittest discover ...")
res = subprocess.run(
    [sys.executable, "-m", "unittest", "discover", "-v", str(tests_dir)],
    check=False,
    cwd=str(root),
    capture_output=True,
    text=True,
)
print("RETURN CODE:", res.returncode)
print("STDOUT:\n", res.stdout)
print("STDERR:\n", res.stderr)
