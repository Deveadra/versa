# Ult / Aerith — Clean Install & Bootstrap

This project installs from `pyproject.toml` (editable install). Wrapper `requirements*.txt` files are optional conveniences.

## Supported Python
- **Text mode / dev:** Python **3.11 or 3.12**
- **Voice mode (local STT):**
  - Python **3.11** installs `openai-whisper`
  - Python **3.12+** uses `faster-whisper` (recommended)

## 0) Clone
```bash
git clone <YOUR_REPO_URL>
cd versa