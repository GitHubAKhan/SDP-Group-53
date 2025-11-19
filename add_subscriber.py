#!/usr/bin/env python3
"""
Quick script to add email subscribers
"""

import json
import os

SUBSCRIBERS_FILE = 'subscribers.json'

def load_subscribers():
    """Load current subscribers"""
    if os.path.exists(SUBSCRIBERS_FILE):
        with open(SUBSCRIBERS_FILE, 'r') as f:
            return json.load(f)
    return {'subscribers': []}

def save_subscribers(data):
    """Save subscribers to file"""
    with open(SUBSCRIBERS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def add_subscriber(email):
    """Add a subscriber"""
    data = load_subscribers()

    if email in data['subscribers']:
        print(f"❌ {email} is already subscribed")
        return False

    data['subscribers'].append(email)
    save_subscribers(data)
    print(f"✅ Added {email} to subscriber list")
    return True

def list_subscribers():
    """Show all subscribers"""
    data = load_subscribers()
    print("\n📧 Current Subscribers:")
    print("=" * 60)
    for i, email in enumerate(data['subscribers'], 1):
        print(f"{i}. {email}")
    print("=" * 60)
    print(f"Total: {len(data['subscribers'])} subscriber(s)\n")

def main():
    print("=" * 60)
    print("UConn Quant - Subscriber Management")
    print("=" * 60)
    print()

    # Show current subscribers
    list_subscribers()

    # Ask for email
    email = input("Enter email to add (or press Enter to quit): ").strip()

    if not email:
        print("No email entered. Exiting.")
        return

    # Validate email
    if '@' not in email or '.' not in email:
        print("❌ Invalid email format")
        return

    # Add subscriber
    add_subscriber(email)

    # Show updated list
    list_subscribers()

if __name__ == '__main__':
    main()
