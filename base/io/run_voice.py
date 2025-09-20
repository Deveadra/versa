
from base.agents.orchestrator import Orchestrator
from base.io.voice import record_audio, transcribe, speak
from base.memory.store import init_db, MemoryStore
from database.sqlite import SQLiteConn

if __name__ == "__main__":
    orch = Orchestrator()
    orch.ingest_bootstrap()
    print("Ultron (voice) ready. Say 'exit' to quit.")

    try:
        while True:
            # Record & transcribe
            path = record_audio(duration=5)
            msg = transcribe(path)
            if not msg:
                continue
            print("You:", msg)

            if msg.lower() in {"exit", "quit"}:
                break

            # Handle command (same logic as run.py)
            if msg.lower() == "facts":
                facts = orch.store.list_facts()
                reply = "Here are your facts: " + "; ".join(f"{k}: {v}" for k, v in facts)
                speak(reply)
                continue

            if msg.lower().startswith("forget "):
                topic = msg.split(" ", 1)[1]
                n = orch.store.forget(topic)
                speak(f"I forgot {n} entries about {topic}.")
                continue

            if msg.lower().startswith("kg "):
                entity = msg.split(" ", 1)[1]
                relations = orch.kg_store.query_relations(entity)
                if relations:
                    reply = "; ".join(f"{src} {rel} {tgt}" for src, rel, tgt, _, _, _ in relations)
                else:
                    reply = f"I don't know any relations for {entity}." 
                speak(reply)
                continue

            # Default chat
            reply = orch.handle_user(msg)
            speak(reply)

    finally:
        orch.shutdown()
