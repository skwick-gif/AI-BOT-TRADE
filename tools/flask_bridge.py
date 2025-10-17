"""
Simple REST Bridge using Flask
"""

from flask import Flask, request, jsonify
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

app = Flask(__name__)

# Initialize service lazily
ibkr_service = None

def get_ibkr_service():
    global ibkr_service
    if ibkr_service is None:
        from services.ibkr_adapter_service import IBKRAdapterService
        from core.config_manager import ConfigManager
        config = ConfigManager()
        ibkr_service = IBKRAdapterService(config.ibkr)
    return ibkr_service

@app.route('/health')
def health():
    """Health check"""
    return jsonify({"status": "ok"})

@app.route('/connect', methods=['POST'])
def connect():
    """Connect to IBKR"""
    try:
        data = request.get_json()
        host = data.get('host', '127.0.0.1')
        port = data.get('port', 7497)
        client_id = data.get('clientId', 1)

        service = get_ibkr_service()
        service.config.host = host
        service.config.port = port
        service.config.client_id = client_id

        success = service.connect()
        return jsonify({"connected": success})
    except Exception as e:
        return jsonify({"connected": False, "error": str(e)}), 400

@app.route('/account')
def get_account():
    """Get account summary"""
    try:
        service = get_ibkr_service()
        if not service.is_connected():
            return jsonify({"error": "Not connected to IBKR"}), 400

        account = service.get_account_info()
        return jsonify(account)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/portfolio')
def get_portfolio():
    """Get portfolio positions"""
    try:
        service = get_ibkr_service()
        if not service.is_connected():
            return jsonify({"error": "Not connected to IBKR"}), 400

        portfolio = service.get_positions()
        return jsonify(portfolio)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"Error starting Flask: {e}")
        import traceback
        traceback.print_exc()