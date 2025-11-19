#!/usr/bin/env python3
"""
Subscription Manager - Handles email subscription add/remove
"""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

SUBSCRIBERS_FILE = 'subscribers.json'


def load_subscribers():
    """Load subscribers from file"""
    if os.path.exists(SUBSCRIBERS_FILE):
        with open(SUBSCRIBERS_FILE, 'r') as f:
            return json.load(f)
    return {'subscribers': []}


def save_subscribers(data):
    """Save subscribers to file"""
    with open(SUBSCRIBERS_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def add_subscriber(email):
    """Add a new subscriber"""
    data = load_subscribers()
    if email not in data['subscribers']:
        data['subscribers'].append(email)
        save_subscribers(data)
        return True, "Successfully subscribed!"
    return False, "Email already subscribed"


def remove_subscriber(email):
    """Remove a subscriber"""
    data = load_subscribers()
    if email in data['subscribers']:
        data['subscribers'].remove(email)
        save_subscribers(data)
        return True, "Successfully unsubscribed"
    return False, "Email not found"


class SubscriptionHandler(BaseHTTPRequestHandler):
    """HTTP handler for subscription requests"""

    def do_OPTIONS(self):
        """Handle preflight CORS requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        """Handle subscription requests"""
        if self.path == '/subscribe':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = urllib.parse.parse_qs(post_data.decode('utf-8'))

            email = params.get('email', [''])[0]

            if email:
                success, message = add_subscriber(email)
                self.send_response(200 if success else 400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = json.dumps({'success': success, 'message': message})
                self.wfile.write(response.encode())
            else:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = json.dumps({'success': False, 'message': 'Email required'})
                self.wfile.write(response.encode())

        elif self.path == '/unsubscribe':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = urllib.parse.parse_qs(post_data.decode('utf-8'))

            email = params.get('email', [''])[0]

            if email:
                success, message = remove_subscriber(email)
                self.send_response(200 if success else 400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = json.dumps({'success': success, 'message': message})
                self.wfile.write(response.encode())

    def do_GET(self):
        """Handle list subscribers request"""
        if self.path == '/subscribers':
            data = load_subscribers()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        """Override to reduce console spam"""
        pass


if __name__ == '__main__':
    PORT = 8080
    print(f"Starting subscription server on port {PORT}")
    print(f"Subscribers file: {SUBSCRIBERS_FILE}")
    print("\nEndpoints:")
    print(f"  POST http://localhost:{PORT}/subscribe")
    print(f"  POST http://localhost:{PORT}/unsubscribe")
    print(f"  GET  http://localhost:{PORT}/subscribers")
    print("\nPress Ctrl+C to stop\n")

    server = HTTPServer(('localhost', PORT), SubscriptionHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        server.shutdown()
