
### `MemoryStore.add_event()`, subscribers are called synchronously in the same thread that called `add_event()`:

```bash
for cb in self._subscribers:
    cb(...)
```

The background thread you start is only for `_embed_and_store_vector(...)`, and that thread does not call subscribers.

So there is nothing to replace for subscribers right now. The reason I raised it is future-proofing: if you ever move subscriber notification into a thread, SQLite + callbacks can get messy fast.

