
def run_text(orch):
  print("Ultron (text) ready. Type 'exit' to quit. Commands: facts, forget <topic>, kg <entity>")
  while True:
    msg = input("You: ").strip()
    if msg.lower() in {"exit", "quit"}:
      break


    if msg.lower() == "facts":
      for k, v in orch.store.list_facts():
        print(f" - {k}: {v}")
      continue


    if msg.lower().startswith("forget "):
      topic = msg.split(" ", 1)[1]
      n = orch.store.forget(topic)
      print(f"Ultron: forgot {n} entries containing '{topic}'.")
      continue


    if msg.lower().startswith("kg "):
      entity = msg.split(" ", 1)[1]
      relations = orch.kg_store.query_relations(entity)
      if relations:
        for src, rel, tgt in relations:
          print(f"Ultron KG: {src} —[{rel}]→ {tgt}")
      else:
        print(f"Ultron KG: no relations found for '{entity}'.")
      continue


  reply = orch.handle_user(msg)
  print("Ultron:", reply)