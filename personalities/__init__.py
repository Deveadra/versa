from personalities.jarvis.lines import JARVIS_PERSONALITY
from personalities.ultron.lines import ULTRON_PERSONALITY

PERSONALITIES = {
    "jarvis": JARVIS_PERSONALITY,
    "ultron": ULTRON_PERSONALITY
}

# Default active personality is now Ultron
CURRENT_PERSONALITY = PERSONALITIES["ultron"]

PERSONALITY_COMMANDS = {
    "switch to jarvis": "jarvis",
    "switch to ultron": "ultron"
}
