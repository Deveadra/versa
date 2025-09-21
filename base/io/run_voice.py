
from base.agents.orchestrator import Orchestrator
from base.io.voice import record_audio, transcribe, speak

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
                if facts:
                    reply = "Here are your facts: " + "; ".join(f"{k}: {v}" for k, v in facts)
                else:
                    reply = "You have no stored facts."
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
                    # if relations are 3-tuples
                    reply = "; ".join(
                        f"{r[0]} {r[1]} {r[2]}" for r in relations
                    )

                    # if they are 6-tuples, use: for src, rel, tgt, *_ in relations
                else:
                    reply = f"I don't know any relations for {entity}."
                speak(reply)
                continue


            # Default chat
            reply = orch.handle_user(msg)
            speak(reply)

    finally:
        orch.shutdown()
