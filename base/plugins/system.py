import psutil
from base.core.stylizer import stylize_response

def get_system_stats(personality=None, mode="default"):
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    data = {"cpu": cpu, "mem": mem}

    if personality:
        return stylize_response(personality, mode, "system", data)

    return f"CPU: {cpu}%, Memory: {mem}%"
