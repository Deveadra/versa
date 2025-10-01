#!/usr/bin/env python3
"""
apply_orchestrator_feedback.py
Safely patch base/agents/orchestrator.py to add feedback/tone wiring.
Creates a backup file orchestrator.py.bak before changing.
Run from repo root: python apply_orchestrator_feedback.py
"""

import re
from pathlib import Path
import shutil
import sys

ROOT = Path(".")
TARGET = ROOT / "base" / "agents" / "orchestrator.py"
if not TARGET.exists():
    print("ERROR: orchestrator.py not found at:", TARGET)
    sys.exit(1)

bak = TARGET.with_suffix(".py.bak")
shutil.copy2(TARGET, bak)
print("Backup created:", bak)

txt = TARGET.read_text(encoding="utf-8", errors="ignore")

# 1) Add imports for Feedback, quick_polarity, ToneAdapter after existing composer/retriever import if present
if "from base.learning.feedback import Feedback" not in txt:
    anchor = "from base.llm.prompt_composer import compose_prompt"
    insert = ("from base.llm.prompt_composer import compose_prompt\n"
              "from base.llm.retriever import Retriever\n"
              "from base.learning.feedback import Feedback\n"
              "from base.learning.sentiment import quick_polarity\n"
              "from base.personality.tone_adapter import ToneAdapter\n")
    if anchor in txt:
        txt = txt.replace(anchor, insert, 1)
        print("Inserted imports after composer anchor.")
    else:
        # try alternate anchor
        anchor2 = "from base.llm.retriever import Retriever"
        if anchor2 in txt:
            txt = txt.replace(anchor2, insert, 1)
            print("Inserted imports after retriever anchor.")
        else:
            # fallback: put near top after module docstring or first import block
            m = re.search(r'(^\s*(?:import |from ).+\n)+', txt, flags=re.M)
            if m:
                pos = m.end()
                txt = txt[:pos] + "\n" + insert + txt[pos:]
                print("Inserted imports after top import block.")
            else:
                txt = insert + txt
                print("Prepended imports at file start (fallback).")

# 2) Enhance __init__ block: attempt to find existing block where PersonaPrimer is instantiated
if "self.primer = PersonaPrimer" in txt and "self.policy_by_usage_id" not in txt:
    # Insert the feedback/tone initialization after the line that sets self.primer
    txt = txt.replace(
        "self.primer = PersonaPrimer(self.profile_mgr, self.miner, self.db)",
        "self.primer = PersonaPrimer(self.profile_mgr, self.miner, self.db)\n"
        "            # optional feedback / tone components\n"
        "            try:\n"
        "                self.feedback = Feedback(self.db)\n"
        "            except Exception:\n"
        "                self.feedback = None\n"
        "            try:\n"
        "                profile = self.profile_mgr.load_profile() if self.profile_mgr else {}\n"
        "            except Exception:\n"
        "                profile = {}\n"
        "            try:\n"
        "                self.tone_adapter = ToneAdapter(profile)\n"
        "            except Exception:\n"
        "                self.tone_adapter = None\n"
        "            # mapping usage_id -> policy_id for async feedback attribution\n"
        "            self.policy_by_usage_id = {}\n"
    )
    print("Patched __init__ after PersonaPrimer instantiation.")
else:
    # fallback: try to insert near a previously added initial block 'self.primer = None' or after 'self.brain = Brain()'
    if "self.primer = None" in txt and "self.policy_by_usage_id" not in txt:
        txt = txt.replace("self.primer = None",
                          "self.primer = None\n            self.feedback = None\n            self.tone_adapter = None\n            self.policy_by_usage_id = {}\n")
        print("Patched fallback __init__ area (primer None).")
    elif "self.brain = Brain()" in txt and "self.policy_by_usage_id" not in txt:
        txt = txt.replace("self.brain = Brain()", "self.brain = Brain()\n" +
                          "        # learning & feedback components (inserted)\n" +
                          "        try:\n" +
                          "            self.feedback = Feedback(self.db)\n" +
                          "        except Exception:\n" +
                          "            self.feedback = None\n" +
                          "        try:\n" +
                          "            profile = {}\n" +
                          "        except Exception:\n" +
                          "            profile = {}\n" +
                          "        try:\n" +
                          "            self.tone_adapter = ToneAdapter(profile)\n" +
                          "        except Exception:\n" +
                          "            self.tone_adapter = None\n" +
                          "        self.policy_by_usage_id = {}\n")
        print("Inserted feedback init after self.brain = Brain() (fallback).")
    else:
        print("Warning: could not find suitable __init__ insertion point - you will need to add initialization manually.")
        # continue; we'll still try the other edits

# 3) Replace the call to LLM: detect a call to self.brain.ask_brain(...prompt...) and replace it with policy selection + composer + mapping
if "reply = self.brain.ask_brain(SYSTEM_PROMPT, prompt)" in txt and "self.policy_by_usage_id" in txt:
    replacement = (
        "# select tone policy (if available) and record it so we can attribute feedback later\n"
        "        try:\n"
        "            policy = self.tone_adapter.choose_policy() if getattr(self, \"tone_adapter\", None) else None\n"
        "            policy_id = policy[\"id\"] if policy else None\n"
        "            # stash last policy for immediate use (and mapping by usage_id later)\n"
        "            self.last_policy_id = policy_id\n"
        "        except Exception:\n"
        "            policy = None\n"
        "            policy_id = None\n"
        "            self.last_policy_id = None\n\n"
        "        # compose final prompt using the composer (persona + memories + extra_context)\n"
        "        prompt = compose_prompt(SYSTEM_PROMPT, user_text, persona_text=persona_text, memories=memories, extra_context=kg_context, top_k_memories=3)\n"
        "        # optionally: you may inject policy instructions into SYSTEM_PROMPT or extra_context based on policy here\n"
        "        reply = self.brain.ask_brain(SYSTEM_PROMPT, prompt)\n\n"
        "        # if we logged a usage for this outgoing reply, attach mapping usage_id -> policy_id so feedback can credit the bandit\n"
        "        try:\n"
        "            if hasattr(self, \"last_usage_id\") and getattr(self, \"last_usage_id\", None):\n"
        "                uid = self.last_usage_id\n"
        "                if policy_id:\n"
        "                    try:\n"
        "                        self.policy_by_usage_id[uid] = policy_id\n"
        "                    except Exception:\n"
        "                        pass\n"
        "        except Exception:\n"
        "            pass\n"
    )
    txt = txt.replace("reply = self.brain.ask_brain(SYSTEM_PROMPT, prompt)", replacement)
    print("Replaced LLM call with policy selection + composer + mapping.")
else:
    print("Warning: did not find exact 'reply = self.brain.ask_brain(SYSTEM_PROMPT, prompt)' string; you may need to edit manually to integrate policy selection.")

# 4) Insert the two methods after __init__ end. Find insertion point: after def __init__ block (look for next 'def ' after it)
if "def ask_confirmation_if_unsure" not in txt:
    m = re.search(r"class\s+Orchestrator\b.*?def\s+__init__\s*\([^)]*\)\s*:\s*", txt, flags=re.S)
    if m:
        # locate end of __init__ by finding the next "\n\s*def\s" after m.end()
        rest = txt[m.end():]
        m2 = re.search(r"\n\s*def\s+", rest)
        if m2:
            insert_pos = m.end() + m2.start()
        else:
            # fallback: insert near the end of the class header region
            insert_pos = m.end()
        methods = """

    def ask_confirmation_if_unsure(self, suggestion: str, confidence: float, usage_id: int = None):
        \"\"\"Return a short confirmation prompt dict if confidence is below threshold.
        Caller should send this to the user via the REPL/UI and then call record_user_feedback with the response.
        \"\"\"
        try:
            if confidence is None:
                return None
            threshold = 0.60
            if confidence < threshold:
                q = f"I can {suggestion}. Did I get that right?"
                return {"ask_user": q, "usage_id": usage_id}
            return None
        except Exception:
            try:
                logger.exception("ask_confirmation_if_unsure failed")
            except Exception:
                pass
            return None

    def record_user_feedback(self, usage_id: int, text: str):
        \"\"\"Record user feedback for a given usage_id, update feedback_events, and reward the tone adapter.\"\"\"
        try:
            s = quick_polarity(text)
            kind = "confirm" if s > 0.2 else "dislike" if s < -0.2 else "note"
            if getattr(self, 'feedback', None):
                try:
                    self.feedback.record(usage_id, kind, text)
                except Exception:
                    try:
                        logger.exception("Failed to record feedback")
                    except Exception:
                        pass
            # reward tone adapter via stored mapping (policy_by_usage_id)
            try:
                pid = None
                if hasattr(self, 'policy_by_usage_id') and usage_id in self.policy_by_usage_id:
                    pid = self.policy_by_usage_id.get(usage_id)
                # fallback to last_policy_id if nothing stored
                if not pid:
                    pid = getattr(self, 'last_policy_id', None)
                if getattr(self, 'tone_adapter', None) and pid is not None:
                    try:
                        self.tone_adapter.reward(pid, s)
                    except Exception:
                        try:
                            logger.exception("Failed to reward tone adapter")
                        except Exception:
                            pass
            except Exception:
                try:
                    logger.exception("Error while rewarding tone_adapter")
                except Exception:
                    pass
        except Exception:
            try:
                logger.exception("record_user_feedback failed")
            except Exception:
                pass

"""
        txt = txt[:insert_pos] + methods + txt[insert_pos:]
        print("Inserted ask_confirmation_if_unsure and record_user_feedback methods.")
    else:
        print("Could not locate Orchestrator.__init__; manual insertion required for methods.")
else:
    print("Methods already present; skipping insertion.")

# Write back file
TARGET.write_text(txt, encoding="utf-8")
print("Patched file written:", TARGET)
print("Done. Please run your tests and verify changes. If anything looks off, restore backup:", bak)
