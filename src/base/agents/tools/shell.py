import subprocess


def run_shell(instruction: str) -> str:
    try:
        # Extract command after colon or last sentence
        parts = instruction.split(":")
        command = parts[-1].strip()
        result = subprocess.run(command, check=False, shell=True, text=True, capture_output=True)
        return result.stdout or result.stderr
    except Exception as e:
        return f"[ShellError] {e}"
