#!/usr/bin/env python3
"""
AWS Lambda Handler — Monthly Momentum Rebalance

Entry point for the AWS Lambda function that runs the monthly rebalance.
Triggered by a CloudWatch Events rule on the first trading day of each month.

Environment variables required (set in Lambda configuration):
    S3_BUCKET         — S3 bucket name for all data and results
    ALPACA_API_KEY    — Alpaca paper trading API key
    ALPACA_API_SECRET — Alpaca paper trading secret key
    FRED_API_KEY      — FRED API key for macro data

Deduplication: writes a lock file to S3 after each run so the cron
(which fires on every weekday in the first 7 days of the month) only
executes the rebalance once per calendar month.
"""

import os
import json
import sys
import traceback
from datetime import datetime, date

# Add project root to path so imports work inside Lambda
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _get_s3_client():
    import boto3
    return boto3.client("s3")


def _already_ran_this_month(s3, bucket):
    """Return True if a lock file for the current month exists in S3."""
    today = date.today()
    lock_key = f"locks/rebalance_{today.year}_{today.month:02d}.json"
    try:
        s3.head_object(Bucket=bucket, Key=lock_key)
        return True, lock_key
    except Exception:
        return False, lock_key


def _write_lock_file(s3, bucket, lock_key, trades_count, dry_run):
    """Write a lock file to S3 so we don't double-fire this month."""
    payload = {
        "ran_at": datetime.utcnow().isoformat() + "Z",
        "dry_run": dry_run,
        "trades_executed": trades_count,
    }
    s3.put_object(
        Bucket=bucket,
        Key=lock_key,
        Body=json.dumps(payload),
        ContentType="application/json",
    )


def _set_credentials_from_env():
    """
    Ensure Alpaca and FRED keys are set as environment variables.
    In Lambda, these come from the function's environment configuration.
    """
    required = ["ALPACA_API_KEY", "ALPACA_API_SECRET", "FRED_API_KEY", "S3_BUCKET"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Set these in the Lambda function's Configuration > Environment variables."
        )


def handler(event, context):
    """
    AWS Lambda entry point.

    Event payload (optional):
        {
            "dry_run": true      # Preview trades without executing (default: false)
            "force": true        # Ignore monthly lock and run anyway (default: false)
        }
    """
    print(f"[Lambda] Triggered at {datetime.utcnow().isoformat()}Z")
    print(f"[Lambda] Event: {json.dumps(event)}")

    # ------------------------------------------------------------------ #
    # Validate environment
    # ------------------------------------------------------------------ #
    try:
        _set_credentials_from_env()
    except EnvironmentError as e:
        print(f"[Lambda] ERROR: {e}")
        return {
            "statusCode": 500,
            "body": str(e),
        }

    bucket = os.environ["S3_BUCKET"]
    dry_run = event.get("dry_run", False)
    force = event.get("force", False)

    s3 = _get_s3_client()

    # ------------------------------------------------------------------ #
    # Deduplication: only run once per calendar month
    # ------------------------------------------------------------------ #
    if not force:
        already_ran, lock_key = _already_ran_this_month(s3, bucket)
        if already_ran:
            msg = f"Rebalance already ran for {date.today().strftime('%Y-%m')}. Skipping."
            print(f"[Lambda] {msg}")
            return {"statusCode": 200, "body": msg}
    else:
        today = date.today()
        lock_key = f"locks/rebalance_{today.year}_{today.month:02d}.json"
        print("[Lambda] Force flag set — ignoring monthly lock.")

    # ------------------------------------------------------------------ #
    # Run the full monthly rebalance pipeline
    # ------------------------------------------------------------------ #
    print(f"[Lambda] Starting monthly rebalance (dry_run={dry_run})...")
    try:
        from execution.scheduler import run_monthly_rebalance
        results = run_monthly_rebalance(dry_run=dry_run)
        trades_count = len(results) if results else 0

        print(f"[Lambda] Rebalance complete. Trades: {trades_count}")

        # Write lock file so cron doesn't fire again this month
        if not dry_run:
            _write_lock_file(s3, bucket, lock_key, trades_count, dry_run)
            print(f"[Lambda] Lock file written: s3://{bucket}/{lock_key}")

        # Write a summary to S3 for the frontend to display
        _write_run_summary(s3, bucket, results, dry_run)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "status": "success",
                "date": date.today().isoformat(),
                "dry_run": dry_run,
                "trades_executed": trades_count,
            }),
        }

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"[Lambda] ERROR during rebalance:\n{error_msg}")

        # Write error to S3 so it's visible
        s3.put_object(
            Bucket=bucket,
            Key=f"logs/error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt",
            Body=error_msg,
            ContentType="text/plain",
        )

        return {
            "statusCode": 500,
            "body": json.dumps({"status": "error", "message": str(e)}),
        }


def _write_run_summary(s3, bucket, results, dry_run):
    """Write a JSON summary of the rebalance to S3 for the frontend."""
    today = date.today()
    summary = {
        "last_rebalance": today.isoformat(),
        "dry_run": dry_run,
        "trades": [],
    }

    if results:
        for r in results:
            if isinstance(r, dict):
                summary["trades"].append(r)

    # Latest run summary (overwritten each month)
    s3.put_object(
        Bucket=bucket,
        Key="results/latest_rebalance.json",
        Body=json.dumps(summary, default=str),
        ContentType="application/json",
    )

    # Dated archive copy
    s3.put_object(
        Bucket=bucket,
        Key=f"results/rebalance_{today.strftime('%Y_%m')}.json",
        Body=json.dumps(summary, default=str),
        ContentType="application/json",
    )
