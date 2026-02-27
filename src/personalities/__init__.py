from personalities.jarvis.lines import JARVIS_PERSONALITY
from personalities.aerith.lines import AERITH_PERSONALITY

PERSONALITIES = {"jarvis": JARVIS_PERSONALITY, "aerith": AERITH_PERSONALITY}

# Default active personality is now Aerith
CURRENT_PERSONALITY = PERSONALITIES["aerith"]

PERSONALITY_COMMANDS = {"switch to jarvis": "jarvis", "switch to aerith": "aerith"}
