def run_text(orch):
    print("Ultron (text) ready. Type 'exit' to quit. Commands: facts, forget <topic>")
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
                print("Ultron facts:")
                for k, v in facts:
                    print(f" - {k}: {v}")
            else:
                print("Ultron: I’m not holding any facts yet.")
            continue

        if msg.lower().startswith("forget "):
            topic = msg.split(" ", 1)[1]
            n = orch.store.forget(topic)
            print(f"Ultron: Removed {n} entries related to '{topic}'.")
            continue

        reply = orch.handle_user(msg)
        print("Ultron:", reply)
