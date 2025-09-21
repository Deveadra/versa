import os
import random
import fnmatch
import subprocess

from base.apps.file_manager_prompts import ASK_FILE_ACTION_VARIANTS, CONFIRM_FILE_VARIANTS, CANCEL_FILE_VARIANTS, CHOOSE_FILE_VARIANTS



file_state = {"query": None, "action": None, "confirm": False, "candidates": [], "chosen": None}


def has_pending():
    return any(v is None for k, v in file_state.items() if k not in ["confirm", "candidates", "chosen"]) or file_state["confirm"] or (file_state["candidates"] and not file_state["chosen"])


def is_file_command(text: str) -> bool:
    return "file" in text.lower() or "document" in text.lower() or "open" in text.lower()


def search_files(query, base_path="."):
    matches = []
    for root, _, files in os.walk(base_path):
        for filename in fnmatch.filter(files, f"*{query}*"):
            matches.append(os.path.join(root, filename))
    return matches


def open_file(path):
    try:
        if os.name == "nt":  # Windows
            os.startfile(path)
        elif os.uname().sysname == "Darwin":  # macOS
            subprocess.run(["open", path])
        else:  # Linux/Unix
            subprocess.run(["xdg-open", path])
        return f"Opened file: {path}"
    except Exception as e:
        return f"Failed to open file: {e}"


def summarize_file(path, lines=5):
    try:
        with open(path, "r", errors="ignore") as f:
            content = f.readlines()
        preview = "".join(content[:lines])
        return f"Summary of {path}:\n{preview}"
    except Exception as e:
        return f"Failed to summarize file: {e}"


def handle(text: str, active_plugins):
    global file_state

    # Handle candidate selection
    if file_state["candidates"] and not file_state["chosen"]:
        try:
            idx = int(text.strip()) - 1
            if 0 <= idx < len(file_state["candidates"]):
                file_state["chosen"] = file_state["candidates"][idx]
                file_state["confirm"] = True
                return None, f"You selected {file_state['chosen']}. Do you want me to {file_state['action']} it?"
            else:
                return None, "Invalid choice. Please provide a valid number."
        except ValueError:
            return None, "Please respond with the number of the file you want."

    # Query step
    if file_state["query"] is None:
        file_state["query"] = text.strip()
        return None, random.choice(ASK_FILE_ACTION_VARIANTS)

    # Action step
    if file_state["action"] is None:
        if any(a in text.lower() for a in ["open", "search", "summarize"]):
            if "open" in text.lower():
                file_state["action"] = "open"
            elif "search" in text.lower():
                file_state["action"] = "search"
            elif "summarize" in text.lower():
                file_state["action"] = "summarize"
            file_state["confirm"] = True
            return None, f"You want me to {file_state['action']} {file_state['query']}?"
        else:
            return None, random.choice(ASK_FILE_ACTION_VARIANTS)

    # Confirmation step
    if file_state["confirm"]:
        if text.lower() in ["yes", "confirm", "do it"]:
            result = None
            if file_state["action"] == "search":
                matches = search_files(file_state["query"])
                result = "Found files:\n" + "\n".join(matches) if matches else "No files found."
            elif file_state["action"] in ["open", "summarize"]:
                matches = search_files(file_state["query"])
                if not matches:
                    result = "No matching files found."
                elif len(matches) == 1:
                    path = matches[0]
                    result = open_file(path) if file_state["action"] == "open" else summarize_file(path)
                else:
                    options = " | ".join(f"{i+1}. {os.path.basename(m)}" for i, m in enumerate(matches[:5]))
                    file_state["candidates"] = matches
                    file_state["confirm"] = False
                    return None, random.choice(CHOOSE_FILE_VARIANTS).format(options=options)
            else:
                result = "Unknown action."

            file_state = {"query": None, "action": None, "confirm": False, "candidates": [], "chosen": None}
            return result, random.choice(CONFIRM_FILE_VARIANTS)
        else:
            file_state = {"query": None, "action": None, "confirm": False, "candidates": [], "chosen": None}
            return "File operation cancelled.", random.choice(CANCEL_FILE_VARIANTS)

    # Initial trigger
    if is_file_command(text):
        file_state = {"query": None, "action": None, "confirm": False, "candidates": [], "chosen": None}
        return None, "Which file are you referring to?"

    return None, None
