from PyQt6.QtCore import QObject, pyqtSignal
from pathlib import Path
import json


class PromptsManager(QObject):
    """Load and watch prompts.json under config/.

    Emits prompts_changed when file is reloaded so UI can pick up changes.
    """
    prompts_changed = pyqtSignal()

    def __init__(self, project_root: Path = None):
        super().__init__()
        if project_root is None:
            project_root = Path(__file__).resolve().parents[3]
        self.project_root = Path(project_root)
        self.prompts_path = self.project_root / "config" / "prompts.json"
        self._data = {}
        self._load()

    def _load(self):
        try:
            if self.prompts_path.exists():
                raw = self.prompts_path.read_text(encoding="utf-8")
                self._data = json.loads(raw)
            else:
                self._data = {}
        except Exception:
            # Keep last-known data on error
            return

    def reload(self):
        prev = json.dumps(self._data, sort_keys=True)
        self._load()
        now = json.dumps(self._data, sort_keys=True)
        if prev != now:
            self.prompts_changed.emit()

    def get_prompt_by_id(self, pid: str) -> str | None:
        try:
            items = self._data.get('prompts', [])
            for p in items:
                if p.get('id') == pid:
                    return p.get('prompt')
        except Exception:
            pass
        return None

    def get_all(self):
        return self._data
