import random

def stylize_response(personality: dict, mode: str, category: str, data: dict) -> str:
    """
    Returns a personality-aware line for plugin results.

    Args:
        personality: currently active personality dictionary
        mode: current mode (default, sarcastic, formal)
        category: plugin category (e.g., "system", "error", "email")
        data: dictionary with plugin-specific info (e.g., {"cpu": 72, "mem": 45})

    Returns:
        A formatted string styled according to personality.
    """

    # Try to get category templates
    templates = personality.get("plugin_lines", {}).get(mode, {}).get(category, [])

    if not templates:
        # Fallback neutral line
        return f"CPU: {data.get('cpu', '?')}%, Memory: {data.get('mem', '?')}%"

    template = random.choice(templates)
    return template.format(**data)
