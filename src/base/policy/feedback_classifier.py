def classify_feedback(text: str) -> str | None:
    t = text.lower().strip()
    if not t:
        return None
    if any(x in t for x in ["thanks", "thank you", "good", "appreciate"]):
        return "thanks"
    if any(x in t for x in ["okay", "fine", "i will", "alright", "sure"]):
        return "acted"
    if any(x in t for x in ["stop", "shut up", "annoying", "nag"]):
        return "angry"
    # fallback: treat as neutral/ignore
    return None
