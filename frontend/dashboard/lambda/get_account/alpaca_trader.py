import requests

class AlpacaTrader:
    def __init__(self, api_key=None, api_secret=None, paper=True):
        """Initialize Alpaca trader."""
        self.api_key = api_key
        self.api_secret = api_secret
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API credentials required")
        
        if paper:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
        
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret,
            'Content-Type': 'application/json'
        }
    
    def get_account(self):
        """Get account information."""
        response = requests.get(
            f"{self.base_url}/v2/account",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_positions(self):
        """Get current positions."""
        response = requests.get(
            f"{self.base_url}/v2/positions",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()