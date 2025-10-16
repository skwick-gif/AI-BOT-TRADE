from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QGroupBox, QMessageBox
)
from PyQt6.QtCore import Qt

from services.ibkr_adapter_service import IBKRAdapterService
from typing import Optional
from core.config_manager import ConfigManager


class ConnectionWizardDialog(QDialog):
    """A simple connection wizard to inspect status, probe, retry, and connect to IBKR via the bridge."""

    def __init__(self, parent=None, service: Optional[IBKRAdapterService] = None):
        super().__init__(parent)
        self.setWindowTitle("IBKR Connection Wizard")
        self.resize(520, 360)

        self.config = ConfigManager()
        self.service = service or IBKRAdapterService(self.config.ibkr)

        layout = QVBoxLayout(self)

        # Bridge info
        info_box = QGroupBox("Bridge Info")
        info_layout = QGridLayout(info_box)
        info_layout.addWidget(QLabel("Bridge URL:"), 0, 0)
        self.bridge_url_label = QLabel(self.service.base_url)
        self.bridge_url_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info_layout.addWidget(self.bridge_url_label, 0, 1)

        info_layout.addWidget(QLabel("Status:"), 1, 0)
        self.status_label = QLabel("Unknown")
        info_layout.addWidget(self.status_label, 1, 1)
        layout.addWidget(info_box)

        # Endpoint config
        ep_box = QGroupBox("Endpoint")
        ep_layout = QGridLayout(ep_box)
        ep_layout.addWidget(QLabel("Host"), 0, 0)
        self.host_edit = QLineEdit(getattr(self.config.ibkr, 'host', '127.0.0.1'))
        ep_layout.addWidget(self.host_edit, 0, 1)

        ep_layout.addWidget(QLabel("Port"), 1, 0)
        self.port_edit = QLineEdit(str(getattr(self.config.ibkr, 'port', 4002)))
        ep_layout.addWidget(self.port_edit, 1, 1)

        ep_layout.addWidget(QLabel("Client ID"), 2, 0)
        self.client_id_edit = QLineEdit(str(getattr(self.config.ibkr, 'client_id', 1)))
        ep_layout.addWidget(self.client_id_edit, 2, 1)
        layout.addWidget(ep_box)

        # Actions
        btn_row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Status")
        self.refresh_btn.clicked.connect(self.refresh_status)
        btn_row.addWidget(self.refresh_btn)

        self.probe_btn = QPushButton("Probe")
        self.probe_btn.clicked.connect(self.probe)
        btn_row.addWidget(self.probe_btn)

        self.retry_btn = QPushButton("Retry Last Connect")
        self.retry_btn.clicked.connect(self.retry)
        btn_row.addWidget(self.retry_btn)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_ibkr)
        btn_row.addWidget(self.connect_btn)

        btn_row.addStretch()
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.close_btn)

        layout.addLayout(btn_row)

        # Initial status
        self.refresh_status()

    def refresh_status(self):
        try:
            st = self.service.get_status()
            if isinstance(st, dict) and st:
                conn = st.get('connected')
                host = st.get('host')
                port = st.get('port')
                code = st.get('errorCode') or st.get('error_code')
                hint = st.get('hint')
                msg = st.get('errorMessage') or st.get('lastError') or st.get('error_message')
                parts = [f"connected={conn}"]
                if host and port:
                    parts.append(f"endpoint={host}:{port}")
                if code:
                    parts.append(f"code={code}")
                if msg:
                    parts.append(f"msg={msg}")
                if hint:
                    parts.append(f"hint={hint}")
                self.status_label.setText("; ".join(parts))
            else:
                self.status_label.setText("Status unavailable (bridge not responding?)")
        except Exception as e:
            self.status_label.setText(f"Status error: {e}")

    def probe(self):
        try:
            host = self.host_edit.text().strip() or '127.0.0.1'
            try:
                port = int(self.port_edit.text().strip() or '4002')
            except Exception:
                port = 4002
            res = self.service.probe(host, port, timeout_ms=800)
            reach = res.get('reachable')
            lat = res.get('latencyMs') or res.get('latency')
            code = res.get('errorCode')
            err = res.get('errorMessage') or res.get('error')
            msg = f"Probe {host}:{port}: reachable={reach} latencyMs={lat}"
            if code or err:
                msg += f" ({code or ''} {err or ''})"
            QMessageBox.information(self, "Probe", msg)
        except Exception as e:
            QMessageBox.warning(self, "Probe Error", str(e))

    def retry(self):
        try:
            ok = self.service.retry_connect()
            self.refresh_status()
            if ok:
                QMessageBox.information(self, "Retry", "Bridge connected after retry.")
            else:
                QMessageBox.warning(self, "Retry", "Retry failed. Check status for details.")
        except Exception as e:
            QMessageBox.warning(self, "Retry Error", str(e))

    def connect_ibkr(self):
        try:
            # Apply edits to config before calling connect
            host = self.host_edit.text().strip() or '127.0.0.1'
            try:
                port = int(self.port_edit.text().strip() or '4002')
            except Exception:
                port = 4002
            try:
                client_id = int(self.client_id_edit.text().strip() or '1')
            except Exception:
                client_id = 1

            try:
                self.service.config.host = host  # type: ignore[attr-defined]
                self.service.config.port = port  # type: ignore[attr-defined]
                self.service.config.client_id = client_id  # type: ignore[attr-defined]
            except Exception:
                pass

            ok = self.service.connect()
            self.refresh_status()
            if ok:
                QMessageBox.information(self, "Connect", "Connected successfully.")
            else:
                QMessageBox.warning(self, "Connect", "Failed to connect. Check status or try Retry.")
        except Exception as e:
            QMessageBox.critical(self, "Connect Error", str(e))
