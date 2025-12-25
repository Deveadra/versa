def read_file(step: str) -> str:
    try:
        filename = step.split()[-1]
        with open(filename) as f:
            return f.read()[:3000]  # Limit output
    except Exception as e:
        return f"[FileReadError] {e}"


def write_file(step: str) -> str:
    try:
        # For simplicity, assume filename and content are in the prompt
        if "::" in step:
            filename, content = step.split("::")
            filename = filename.strip().split()[-1]
        else:
            return "[WriteError] No content provided"
        with open(filename, "w") as f:
            f.write(content.strip())
        return f"Wrote to {filename}"
    except Exception as e:
        return f"[FileWriteError] {e}"
