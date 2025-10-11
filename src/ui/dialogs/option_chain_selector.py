from __future__ import annotations
from typing import Optional
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QLineEdit, QDialogButtonBox, QComboBox

class OptionChainSelectorDialog(QDialog):
    """Minimal chain selector dialog (placeholder). Later we can add expiry/strikes pickers."""
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Option Chain")
        layout = QVBoxLayout(self)
        form = QFormLayout()
        # Combos with editable fallback
        self.expiry_combo = QComboBox(self)
        self.expiry_combo.setEditable(True)
        self.k_combos: list[QComboBox] = []
        # Default to two leg inputs; caller can reconfigure via set_choices(leg_count=...)
        form.addRow("Expiry:", self.expiry_combo)
        # placeholder leg combos created by configure_leg_count
        self.configure_leg_count(2)
        for idx, combo in enumerate(self.k_combos, start=1):
            form.addRow(f"K{idx}:", combo)
        layout.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def values(self) -> dict:
        out = {"E1": (self.expiry_combo.currentText() or "").strip()}
        for i, combo in enumerate(self.k_combos, start=1):
            out[f"K{i}"] = self._to_float(combo.currentText())
        return out

    def set_choices(self, expirations: list[str] | None, strikes: list[float] | None, leg_count: int = 2):
        """Populate choices and configure number of leg inputs.

        expirations: list of expiry strings
        strikes: list of strike floats
        leg_count: how many K inputs to show (mapped to K1..Kn)
        """
        try:
            # configure leg count first
            self.configure_leg_count(max(1, int(leg_count)))
            if expirations:
                self.expiry_combo.clear()
                self.expiry_combo.addItems([str(e) for e in expirations])
            if strikes:
                texts = [str(int(s)) if float(s).is_integer() else str(s) for s in strikes]
                for combo in self.k_combos:
                    combo.clear()
                    combo.addItems(texts)
        except Exception:
            pass

    def configure_leg_count(self, n: int) -> None:
        """Ensure there are exactly n combo boxes for leg strikes."""
        try:
            n = max(1, int(n))
        except Exception:
            n = 2
        # remove existing if more than needed
        if len(self.k_combos) > n:
            self.k_combos = self.k_combos[:n]
            return
        # add missing combos
        while len(self.k_combos) < n:
            combo = QComboBox(self)
            combo.setEditable(True)
            self.k_combos.append(combo)

    def _to_float(self, s: str) -> float:
        try:
            return float(s.strip())
        except Exception:
            return 0.0
