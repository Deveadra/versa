from base.agents.orchestrator import Orchestrator
from base.io.interface import launch_interface

if __name__ == "__main__":
    orch = Orchestrator()
    orch.ingest_bootstrap()
    print("Ultron ready. Type 'exit' to quit. Commands: facts, forget <topic>, kg <entity>")
    try:
        launch_interface(orch)
    except:
        while True:
            msg = input("You: ").strip()
            if msg.lower() in {"exit", "quit"}:
                break

            # List all facts
            if msg.lower() == "facts":
                for k, v in orch.store.list_facts():
                    print(f" - {k}: {v}")
                continue

            # Forget by keyword
            if msg.lower().startswith("forget "):
                topic = msg.split(" ", 1)[1]
                n = orch.store.forget(topic)
                print(f"Ultron: forgot {n} entries containing '{topic}'.")
                continue

            # Knowledge Graph query
            if msg.lower().startswith("kg "):
                entity = msg.split(" ", 1)[1]
                relations = orch.kg_store.query_relations(entity)
                if relations:
                    for src, rel, tgt, conf, vfrom, vto in relations:
                        print(f"Ultron KG: {src} —[{rel}]→ {tgt}")
                else:
                    print(f"Ultron KG: no relations found for '{entity}'.")
                continue
                        # Run self-improvement (manual trigger)
            if msg.lower() == "self-improve":
                try:
                    orch._job_self_improvement()
                    print("Ultron: self-improvement job executed.")
                except Exception as e:
                    print(f"Ultron: self-improvement failed: {e}")
                continue

            # Default flow (chat)
            reply = orch.handle_user(msg)
            print("Ultron:", reply)
    finally:
        orch.shutdown()
