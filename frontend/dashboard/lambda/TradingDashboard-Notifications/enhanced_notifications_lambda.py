import json
import boto3
import smtplib
import hmac
import hashlib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urlencode
import os
import urllib3

# ==========================
# CONFIG (from environment)
# ==========================

BUCKET_NAME = "trading-dashboard-sdteam-53"

EMAIL_ADDRESS = os.environ["EMAIL_ADDRESS"]
EMAIL_PASSWORD = os.environ["EMAIL_PASSWORD"]

ALPACA_API_KEY = os.environ["ALPACA_API_KEY"]
ALPACA_API_SECRET = os.environ["ALPACA_API_SECRET"]

UNSUBSCRIBE_SECRET = os.environ["UNSUBSCRIBE_SECRET"]
UNSUBSCRIBE_BASE_URL = os.environ["UNSUBSCRIBE_BASE_URL"]  # your Function URL

s3 = boto3.client("s3")

# ==========================
# UNSUBSCRIBE HELPERS
# ==========================

def generate_unsubscribe_token(email: str) -> str:
    return hmac.new(
        UNSUBSCRIBE_SECRET.encode(),
        email.lower().encode(),
        hashlib.sha256
    ).hexdigest()

def get_unsubscribe_url(email: str) -> str:
    token = generate_unsubscribe_token(email)
    params = urlencode({"email": email.lower(), "token": token})
    return f"{UNSUBSCRIBE_BASE_URL}?{params}"

def verify_token(email: str, token: str) -> bool:
    expected = generate_unsubscribe_token(email.lower())
    return hmac.compare_digest(expected, token)

# ==========================
# UNSUBSCRIBE HANDLER
# ==========================

def handle_unsubscribe(event):
    params = event.get("queryStringParameters") or {}
    email = params.get("email", "").strip().lower()
    token = params.get("token", "").strip()

    def html_response(status, message):
        return {
            "statusCode": status,
            "headers": {"Content-Type": "text/html"},
            "body": f"""
                <html>
                <body style="font-family:Arial,sans-serif; text-align:center;
                             padding:60px; background:#0C2340; color:white;">
                    <div style="background:#00205B; display:inline-block;
                                padding:40px 60px; border-radius:12px;">
                        <h1 style="margin-top:0;">UConn Quant</h1>
                        {message}
                    </div>
                </body>
                </html>
            """
        }

    if not email or not token:
        return html_response(400, "<p>⚠️ Invalid unsubscribe link.</p>")

    if not verify_token(email, token):
        return html_response(403, "<p>⛔ This link is invalid or has been tampered with.</p>")

    # Load subscribers
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key="subscribers.json")
        data = json.loads(response["Body"].read())
    except Exception as e:
        print(f"S3 read error: {e}")
        return html_response(500, "<p>Something went wrong. Please try again later.</p>")

    original = data.get("subscribers", [])
    updated = [e for e in original if e.lower() != email]

    if len(updated) == len(original):
        return html_response(200, f"<p>{email} was not found in our subscriber list.</p>")

    # Write back
    try:
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key="subscribers.json",
            Body=json.dumps({"subscribers": updated}),
            ContentType="application/json"
        )
    except Exception as e:
        print(f"S3 write error: {e}")
        return html_response(500, "<p>Something went wrong. Please try again later.</p>")

    return html_response(200, f"""
        <h2>✅ Unsubscribed</h2>
        <p><strong>{email}</strong> has been removed from all UConn Quant emails.</p>
    """)

# ==========================
# LOAD SUBSCRIBERS FROM S3
# ==========================

def load_subscribers():
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key="subscribers.json")
        data = json.loads(response["Body"].read())
        return data.get("subscribers", [])
    except Exception as e:
        print(f"Error loading subscribers: {e}")
        return []

# ==========================
# FETCH PORTFOLIO DATA
# ==========================

def get_portfolio_summary():
    http = urllib3.PoolManager()
    response = http.request(
        'GET',
        'https://paper-api.alpaca.markets/v2/account',
        headers={
            'APCA-API-KEY-ID': ALPACA_API_KEY,
            'APCA-API-SECRET-KEY': ALPACA_API_SECRET
        }
    )
    account = json.loads(response.data.decode('utf-8'))

    equity = float(account['equity'])
    last_equity = float(account['last_equity'])
    change = equity - last_equity
    percent_change = (change / last_equity) * 100

    positions = get_positions()
    sorted_positions = sorted(
        positions,
        key=lambda x: float(x["unrealized_plpc"]),
        reverse=True
    )

    return equity, change, percent_change, sorted_positions[:3], sorted_positions[-3:]

# ==========================
# FETCH POSITION DATA
# ==========================

def get_positions():
    http = urllib3.PoolManager()
    response = http.request(
        'GET',
        'https://paper-api.alpaca.markets/v2/positions',
        headers={
            'APCA-API-KEY-ID': ALPACA_API_KEY,
            'APCA-API-SECRET-KEY': ALPACA_API_SECRET
        }
    )
    return json.loads(response.data.decode('utf-8'))

# ==========================
# BUILD EMAIL
# ==========================

def build_email(notification_type, recipient_email):
    equity, change, percent_change, top_gainers, top_losers = get_portfolio_summary()
    unsubscribe_url = get_unsubscribe_url(recipient_email)

    now = datetime.now(ZoneInfo("America/New_York")).strftime(
        "%A, %B %d, %Y — %I:%M %p ET"
    )

    is_up = change >= 0
    color = "#16a34a" if is_up else "#dc2626"
    sign = "+" if is_up else "-"

    equity_str = f"${equity:,.2f}"
    change_str = f"{sign}${abs(change):,.2f}"
    pct_str = f"{sign}{abs(percent_change):,.2f}%"

    footer = f"""
    <tr>
        <td style="background:#f9fafb; padding:15px; text-align:center;
                   font-size:12px; color:#6b7280;">
            UConn Quant — Momentum Based Trading System
            <br><br>
            <a href="{unsubscribe_url}"
               style="color:#9ca3af; font-size:11px; text-decoration:underline;">
                Unsubscribe from these emails
            </a>
        </td>
    </tr>
    """

    if notification_type == "market_open":
        subject = f"Market Open — {pct_str}"

        holdings_html = ""
        for pos in top_gainers[:5]:
            holdings_html += f"""
            <tr>
                <td style="padding:8px 0; font-size:14px;">{pos['symbol']}</td>
                <td style="padding:8px 0; font-size:14px;">{pos['qty']} shares</td>
                <td style="padding:8px 0; font-size:14px; color:#16a34a;">
                    {float(pos['unrealized_plpc'])*100:.2f}%
                </td>
            </tr>
            """

        body = f"""
        <html>
        <body style="margin:0; padding:0; background:#0C2340; font-family:Arial, sans-serif;">
        <table width="100%" cellpadding="0" cellspacing="0" style="background:#0C2340; padding:40px 0;">
        <tr><td align="center">
        <table width="600" cellpadding="0" cellspacing="0"
               style="background:#ffffff; border-radius:12px; overflow:hidden;
                      box-shadow:0 20px 40px rgba(0,0,0,0.25);">
            <tr>
                <td style="background:#00205B; padding:30px; color:white;">
                    <h1 style="margin:0; font-size:24px;">Market Open</h1>
                    <p style="margin:8px 0 0; font-size:14px; opacity:0.9;">
                        Markets are open — here is your portfolio snapshot.
                    </p>
                    <p style="margin:8px 0 0; font-size:12px; opacity:0.8;">{now}</p>
                </td>
            </tr>
            <tr>
                <td style="padding:30px; color:#111827;">
                    <h2 style="margin-top:0;">Portfolio Overview</h2>
                    <div style="background:#f3f4f6; padding:20px; border-radius:8px;">
                        <div style="font-size:13px; color:#4b5563;">Total Equity</div>
                        <div style="font-size:28px; font-weight:bold; margin-top:6px;">
                            {equity_str}
                        </div>
                    </div>
                    <h3 style="margin-top:30px;">Top Holdings</h3>
                    <table width="100%" style="border-collapse:collapse;">
                        <tr style="border-bottom:1px solid #e5e7eb;">
                            <th align="left" style="padding-bottom:8px;">Symbol</th>
                            <th align="left" style="padding-bottom:8px;">Shares</th>
                            <th align="left" style="padding-bottom:8px;">Return</th>
                        </tr>
                        {holdings_html}
                    </table>
                </td>
            </tr>
            {footer}
        </table>
        </td></tr>
        </table>
        </body>
        </html>
        """
        return subject, body, unsubscribe_url

    else:
        subject = f"Markets Closed — {pct_str}"

        gainers_html = ""
        for pos in top_gainers:
            gainers_html += f"""
            <tr>
                <td style="padding:6px 0;">{pos['symbol']}</td>
                <td style="padding:6px 0; color:#16a34a;">
                    {float(pos['unrealized_plpc'])*100:.2f}%
                </td>
            </tr>
            """

        losers_html = ""
        for pos in top_losers:
            losers_html += f"""
            <tr>
                <td style="padding:6px 0;">{pos['symbol']}</td>
                <td style="padding:6px 0; color:#dc2626;">
                    {float(pos['unrealized_plpc'])*100:.2f}%
                </td>
            </tr>
            """

        body = f"""
        <html>
        <body style="margin:0; padding:0; background:#0C2340; font-family:Arial, sans-serif;">
        <table width="100%" cellpadding="0" cellspacing="0" style="background:#0C2340; padding:40px 0;">
        <tr><td align="center">
        <table width="600" cellpadding="0" cellspacing="0"
               style="background:#ffffff; border-radius:12px; overflow:hidden;
                      box-shadow:0 20px 40px rgba(0,0,0,0.25);">
            <tr>
                <td style="background:#00205B; padding:30px; color:white;">
                    <h1 style="margin:0; font-size:24px;">Markets Closed</h1>
                    <p style="margin:8px 0 0; font-size:14px; opacity:0.9;">
                        Here is how your portfolio performed today.
                    </p>
                    <p style="margin:8px 0 0; font-size:12px; opacity:0.8;">{now}</p>
                </td>
            </tr>
            <tr>
                <td style="padding:30px; color:#111827;">
                    <div style="text-align:center; background:#f3f4f6; padding:20px; border-radius:8px;">
                        <div style="font-size:13px; color:#4b5563;">Daily Change</div>
                        <div style="font-size:28px; font-weight:bold; color:{color}; margin-top:6px;">
                            {change_str}
                        </div>
                        <div style="font-size:14px; color:{color}; margin-top:4px;">{pct_str}</div>
                        <div style="margin-top:10px;">Portfolio Value: {equity_str}</div>
                    </div>
                    <h3 style="margin-top:30px;">Top Performers</h3>
                    <table width="100%" style="border-collapse:collapse;">
                        {gainers_html}
                    </table>
                    <h3 style="margin-top:25px;">Worst Performers</h3>
                    <table width="100%" style="border-collapse:collapse;">
                        {losers_html}
                    </table>
                </td>
            </tr>
            {footer}
        </table>
        </td></tr>
        </table>
        </body>
        </html>
        """
        return subject, body, unsubscribe_url

# ==========================
# SEND EMAIL
# ==========================

def send_email(to_email, subject, html_body, unsubscribe_url):
    msg = MIMEMultipart("alternative")
    msg["From"] = f"Trading Dashboard <{EMAIL_ADDRESS}>"
    msg["To"] = to_email
    msg["Subject"] = subject
    msg["List-Unsubscribe"] = f"<{unsubscribe_url}>"
    msg["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"

    text_fallback = (
        f"{subject}\n"
        f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"(Open in an HTML-capable client to view the formatted version.)\n\n"
        f"Unsubscribe: {unsubscribe_url}"
    )

    msg.attach(MIMEText(text_fallback, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)

# ==========================
# SEND NOTIFICATION
# ==========================

def send_notification(notification_type):
    subscribers = load_subscribers()
    if not subscribers:
        print("No subscribers found.")
        return

    for email in subscribers:
        print(f"Sending to {email}")
        subject, body, unsubscribe_url = build_email(notification_type, email)
        send_email(email, subject, body, unsubscribe_url)

# ==========================
# LAMBDA ENTRY POINT
# ==========================

def lambda_handler(event, context):
    # Function URL request (unsubscribe click)
    if "requestContext" in event:
        return handle_unsubscribe(event)

    # EventBridge scheduled trigger
    notification_type = event.get("notification_type", "market_open")
    if notification_type not in ["market_open", "market_close"]:
        return {"statusCode": 400, "body": "Invalid notification type"}

    send_notification(notification_type)
    return {"statusCode": 200, "body": f"{notification_type} notification sent"}