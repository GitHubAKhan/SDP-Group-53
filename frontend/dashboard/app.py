from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from alpaca_trader_full import AlpacaTrader
import requests

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Allow frontend requests

# Serve the dashboard page
@app.route('/')
def index():
    return render_template('dashboard_prog.html')

# API endpoint to get account info
@app.route('/api/account', methods=['POST'])
def get_account():
    data = request.json
    trader = AlpacaTrader(api_key=data['apiKey'], api_secret=data['apiSecret'])
    account = trader.get_account()
    return jsonify(account)

# API endpoint to get positions
@app.route('/api/positions', methods=['POST'])
def get_positions():
    data = request.json
    trader = AlpacaTrader(api_key=data['apiKey'], api_secret=data['apiSecret'])
    positions = trader.get_positions()
    return jsonify(positions)

if __name__ == '__main__':
    app.run(debug=True)
