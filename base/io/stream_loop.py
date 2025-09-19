
def run_stream(orch):
  print("⚠️ Streaming STT/TTS not yet implemented. Falling back to text mode.")
  from .text_loop import run_text
  run_text(orch)