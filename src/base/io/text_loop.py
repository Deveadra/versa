def run_text(orch):
    print("Aerith (text) ready. Type 'exit' to quit. Commands: facts, forget <topic>")
    while True:
        msg = input("You: ").strip()
        if not msg:
            continue
        if msg.lower() in {"exit", "quit"}:
            break

        # Small helper commands
        if msg.lower() == "facts":
            facts = orch.store.list_facts()
            if facts:
                print("Aerith facts:")
                for k, v in facts:
                    print(f" - {k}: {v}")
            else:
                print("Aerith: I’m not holding any facts yet.")
            continue

        if msg.lower().startswith("forget "):
            topic = msg.split(" ", 1)[1]
            n = orch.store.forget(topic)
            print(f"Aerith: Removed {n} entries related to '{topic}'.")
            continue

        if msg.lower() == "self-improve":
            try:
                orch._job_self_improvement()
                print("Aerith: self-improvement job executed.")
            except Exception as e:
                print(f"Aerith: self-improvement failed: {e}")
            continue

        reply = orch.handle_user(msg)
        print("Aerith:", reply)
