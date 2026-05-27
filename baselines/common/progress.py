import json
from pathlib import Path


class ProgressTracker:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.done = set(json.loads(self.path.read_text()))
        else:
            self.done = set()

    def is_done(self, key: str) -> bool:
        return key in self.done

    def mark_done(self, key: str) -> None:
        self.done.add(key)

    def save(self) -> None:
        self.path.write_text(json.dumps(sorted(self.done)))
