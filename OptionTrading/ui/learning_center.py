import os
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QTextEdit, QPushButton, QFileDialog, QLabel

class LearningCenter(QDialog):
    def __init__(self, docs_dir: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Learning Center — Strategies (Markdown)")
        self.resize(820, 600)
        self.docs_dir = docs_dir

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.list = QListWidget()
        self.viewer = QTextEdit(); self.viewer.setReadOnly(True)
        top.addWidget(self.list, 1); top.addWidget(self.viewer, 3)
        root.addLayout(top)

        btns = QHBoxLayout()
        self.btn_open = QPushButton("Open Docs Folder")
        self.btn_reload = QPushButton("Reload")
        btns.addWidget(QLabel("Browse strategy docs and read them in Markdown.")); btns.addStretch(1)
        btns.addWidget(self.btn_open); btns.addWidget(self.btn_reload)
        root.addLayout(btns)

        self.btn_open.clicked.connect(self._open_folder)
        self.btn_reload.clicked.connect(self._load_docs)
        self.list.itemSelectionChanged.connect(self._on_select)

        self._load_docs()

    def _open_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Pick Docs Folder", self.docs_dir)
        if d:
            self.docs_dir = d
            self._load_docs()

    def _load_docs(self):
        self.list.clear()
        if not os.path.isdir(self.docs_dir):
            return
        for name in sorted(os.listdir(self.docs_dir)):
            if name.lower().endswith(".md"):
                self.list.addItem(name)

    def _on_select(self):
        item = self.list.currentItem()
        if not item:
            return
        path = os.path.join(self.docs_dir, item.text())
        try:
            text = open(path, "r", encoding="utf-8").read()
        except Exception as e:
            text = f"# Error\n{e}"
        # Simple markdown -> show as-is (Qt6 QTextEdit doesn't parse MD fully; for richer, use QtWebEngine). 
        # We'll prefix a monospace heading and show text. 
        self.viewer.setPlainText(text)
