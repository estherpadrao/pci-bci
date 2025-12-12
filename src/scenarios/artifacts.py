import os, json, pickle, hashlib
from typing import Any, Optional

def stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(s).hexdigest()

class ArtifactStore:
    """
    File-based cache. Point this at Google Drive in Colab so heavy artifacts persist.
    """
    def __init__(self, root: str):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def _path(self, name: str, key: str) -> str:
        return os.path.join(self.root, f"{name}__{key}.pkl")

    def load(self, name: str, key: str) -> Optional[Any]:
        p = self._path(name, key)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
        return None

    def save(self, name: str, key: str, obj: Any) -> Any:
        p = self._path(name, key)
        with open(p, "wb") as f:
            pickle.dump(obj, f)
        return obj
