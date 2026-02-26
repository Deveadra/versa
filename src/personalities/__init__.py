from personalities.jarvis.lines import JARVIS_PERSONALITY
from personalities.aerith.lines import ULTRON_PERSONALITY

PERSONALITIES = {"jarvis": JARVIS_PERSONALITY, "aerith": ULTRON_PERSONALITY}

# Default active personality is now Aerith
CURRENT_PERSONALITY = PERSONALITIES["aerith"]

PERSONALITY_COMMANDS = {"switch to jarvis": "jarvis", "switch to aerith": "aerith"}
