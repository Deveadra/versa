# base/voice/null_voice.py


class NullVoice:
    def speak(self, text: str):
        pass

    def speak_async(self, text: str):
        pass
