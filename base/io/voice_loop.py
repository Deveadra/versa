
from .voice import record_audio, transcribe, speak


def run_voice(orch):
  print("Ultron (voice) ready. Say 'exit' to quit.")
  while True:
    path = record_audio(duration=5)
    msg = transcribe(path)
    if not msg:
      continue
    print("You:", msg)


    if msg.lower() in {"exit", "quit"}:
      break


    if msg.lower() == "facts":
      facts = orch.store.list_facts()
      reply = "Here are your facts: " + "; ".join(f"{k}: {v}" for k, v in facts)
      speak(reply)
    

    elif msg.lower().startswith("forget "):
      topic = msg.split(" ", 1)[1]
      n = orch.store.forget(topic)
      speak(f"I forgot {n} entries about {topic}.")
    

    elif msg.lower().startswith("kg "):
      entity = msg.split(" ", 1)[1]
      relations = orch.kg_store.query_relations(entity)
    if relations:
      reply = "; ".join(f"{src} {rel} {tgt}" for src, rel, tgt in relations)
    else:
      reply = f"I don't know any relations for {entity}."
      speak(reply)
    continue


  reply = orch.handle_user(msg)
  speak(reply)