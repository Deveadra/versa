class PluginManager:
    def __init__(self):
        self.plugins = {}

    def register(self, name, handler, keywords=None, flow=False):
        """
        Register a plugin.

        Args:
            name: plugin name (string)
            handler: function or module with handle() entrypoint
            keywords: list of trigger keywords
            flow: whether this plugin supports multi-step interactions
        """
        self.plugins[name] = {
            "handler": handler,
            "keywords": keywords or [],
            "flow": flow
        }

    def handle(self, text, plugins, personality=None, mode="default"):
        """
        Dispatch user input to a matching plugin.

        Args:
            text: user input string
            plugins: dict of registered plugins
            personality: active personality dictionary
            mode: current personality mode

        Returns:
            (reply_text, spoken_text) tuple
        """
        for name, plugin in plugins.items():
            for keyword in plugin["keywords"]:
                if keyword in text.lower():
                    handler = plugin["handler"]

                    # If handler is a module with handle(), call that
                    if hasattr(handler, "handle"):
                        return handler.handle(text, personality, mode), None

                    # If handler is a function, try injecting personality/mode
                    try:
                        return handler(personality=personality, mode=mode), None
                    except TypeError:
                        # fallback if function doesn't accept those args
                        return handler(), None

        return None, None
