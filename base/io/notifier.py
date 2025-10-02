
from loguru import logger

class Notifier:
    def __init__(self, mode="console"):
        self.mode = mode

    def notify(self, title: str, message: str):
        if self.mode == "console":
            print(f"[Ultron Notification] {title}: {message}")
        else:
            logger.warning(f"Notifier mode {self.mode} not yet implemented")
