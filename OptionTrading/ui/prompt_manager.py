import json, os
from typing import Dict, Any
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel, QFileDialog, QMessageBox

class PromptManager(QDialog):
    def __init__(self, template_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prompt Templates Manager")
        self.resize(840, 600)
        self.template_path = template_path

        root = QVBoxLayout(self)
        root.addWidget(QLabel("Edit JSON templates for API prompts and output schema enforcement:"))
        self.editor = QTextEdit()
        root.addWidget(self.editor, 1)

        btns = QHBoxLayout()
        self.btn_load = QPushButton("Load")
        self.btn_save = QPushButton("Save")
        self.btn_save_as = QPushButton("Save As...")
        btns.addStretch(1); btns.addWidget(self.btn_load); btns.addWidget(self.btn_save); btns.addWidget(self.btn_save_as)
        root.addLayout(btns)

        self.btn_load.clicked.connect(self._load)
        self.btn_save.clicked.connect(self._save)
        self.btn_save_as.clicked.connect(self._save_as)

        # Load on start
        self._load()

    def _load(self):
        try:
            with open(self.template_path, "r", encoding="utf-8") as f:
                self.editor.setPlainText(f.read())
        except Exception as e:
            self.editor.setPlainText(json.dumps(self._default_templates(), indent=2, ensure_ascii=False))

    def _save(self):
        try:
            data = json.loads(self.editor.toPlainText())
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid JSON: {e}")
            return
        with open(self.template_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2, ensure_ascii=False))
        QMessageBox.information(self, "Saved", "Templates saved.")

    def _save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Templates As", os.path.dirname(self.template_path), "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(self.editor.toPlainText())
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid JSON: {e}")
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2, ensure_ascii=False))
        self.template_path = path
        QMessageBox.information(self, "Saved", f"Templates saved to {path}.")

    def _default_templates(self) -> Dict[str, Any]:
        return {
          "analyze_symbol": {
            "system": "You are Perplexity Finance. Provide options strategy ideas. Respond ONLY in compact JSON.",
            "user": "Analyze {symbol} with this snapshot: {snapshot}. Return JSON: {\"strategies\":[{\"key\":str,\"name\":str,\"blurb\":str,\"confidence\":num,\"expected_profit\":num,\"max_loss\":num,\"success_prob\":num,\"timeframe\":str}]]}",
            "schema": {"strategies": []},
            "temperature": 0.2,
            "max_tokens": 700
          }
        }
