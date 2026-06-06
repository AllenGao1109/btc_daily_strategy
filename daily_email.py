"""Daily position-guidance email for the BTC factor_composite strategy.

Fetches the latest data, computes the multi-band decision guidance, and emails
today's actionable call (HOLD vs REBALANCE, and the position change) for the
configured band, plus the full multi-band table for reference.

Credentials are read from a LOCAL, git-ignored file (email_config.json) or env
vars - never hard-coded, never committed. Use --dry-run to preview without sending.

Run:
    python3 daily_email.py --dry-run        # preview
    python3 daily_email.py                   # fetch latest + send
Intended to be driven daily by launchd (see automation/).
"""

from __future__ import annotations

import argparse
import json
import os
import smtplib
import ssl
import traceback
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from src.config import load_config
from src.data import load_btc_data
from src.guidance import build_guidance, render_html, render_markdown

CONFIG_FILE = Path(__file__).parent / "email_config.json"


def load_email_config() -> dict:
    """Load SMTP/email settings from email_config.json or EMAIL_* env vars."""
    if CONFIG_FILE.exists():
        cfg = json.loads(CONFIG_FILE.read_text())
    else:
        cfg = {
            "smtp_host": os.environ.get("EMAIL_SMTP_HOST", ""),
            "smtp_port": int(os.environ.get("EMAIL_SMTP_PORT", "587")),
            "username": os.environ.get("EMAIL_USERNAME", ""),
            "password": os.environ.get("EMAIL_PASSWORD", ""),
            "sender": os.environ.get("EMAIL_SENDER", ""),
            "recipient": os.environ.get("EMAIL_RECIPIENT", ""),
        }
    missing = [k for k in ("smtp_host", "username", "password", "sender", "recipient") if not cfg.get(k)]
    if missing:
        raise RuntimeError(
            f"Email config incomplete (missing {missing}). Fill email_config.json "
            f"(copy from email_config.example.json) or set EMAIL_* env vars."
        )
    cfg.setdefault("smtp_port", 587)
    return cfg


def compose(config: dict) -> tuple[str, str, str]:
    """Build (subject, text_body, html_body) from the latest guidance."""
    table, meta = build_guidance(config)
    primary_band = float(config["backtest"].get("weight_band", 0.40))
    row = min(table.to_dict("records"), key=lambda r: abs(r["band"] - primary_band))
    target = meta["signal_target"]
    holding = row["now_holding"]
    action = row["action_now"]
    is_trade = action.startswith("REBALANCE")

    flag = "⚠️ 换仓 REBALANCE" if is_trade else "持有 HOLD"
    subject = (f"[BTC策略 {meta['as_of']}] {flag}"
               + (f" → {target:.2f}x (原 {holding:.2f}x)" if is_trade else f" {holding:.2f}x"))

    text_body = "\n".join([
        f"BTC factor_composite - daily guidance ({meta['as_of']})",
        f"BTC close: ${meta['btc_close']:,.0f}",
        "",
        f"== Your band ({row['band']:.2f}, ~{row['trades_per_yr']} trades/yr) ==",
        f"  Model signal target : {target:.2f}x",
        f"  Currently holding   : {holding:.2f}x",
        f"  Days since last trade: {row['days_since_trade']}",
        f"  ACTION TODAY        : {action}",
        "",
        ("  >> A position change is signaled today. <<" if is_trade
         else "  >> No change. Hold. <<"),
        "",
        render_markdown(table, meta),
    ])
    html_body = render_html(table, meta, primary_band=row["band"])
    return subject, text_body, html_body


def _build_message(ecfg: dict, subject: str, text_body: str, html_body: str) -> MIMEMultipart:
    """Assemble a multipart/alternative email (HTML with plain-text fallback)."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = ecfg["sender"]
    msg["To"] = ecfg["recipient"]
    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    return msg


def send_email(ecfg: dict, msg: MIMEMultipart) -> None:
    """Send a prepared multipart email over SMTP (STARTTLS)."""
    ctx = ssl.create_default_context()
    with smtplib.SMTP(ecfg["smtp_host"], int(ecfg["smtp_port"]), timeout=30) as s:
        s.starttls(context=ctx)
        s.login(ecfg["username"], ecfg["password"])
        s.send_message(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="print instead of sending")
    ap.add_argument("--no-reload", action="store_true", help="skip fetching latest data")
    args = ap.parse_args()

    config = load_config(args.config)
    if not args.no_reload:
        try:
            load_btc_data(config, force_reload=True)  # refresh today's bar + caches
        except Exception as exc:
            print(f"[warn] data refresh failed, using cache: {exc}")

    try:
        subject, text_body, html_body = compose(config)
    except Exception:
        subject = "[BTC Strategy] ERROR generating guidance"
        text_body = "The daily guidance job failed:\n\n" + traceback.format_exc()
        html_body = f"<pre>{text_body}</pre>"

    if args.dry_run:
        print("SUBJECT:", subject)
        print("-" * 60)
        print(text_body)
        print("-" * 60, "\n[HTML body generated:", len(html_body), "chars]")
        Path("logs").mkdir(exist_ok=True)
        Path("logs/email_preview.html").write_text(html_body)
        print("HTML preview saved -> logs/email_preview.html")
        return

    ecfg = load_email_config()
    msg = _build_message(ecfg, subject, text_body, html_body)
    send_email(ecfg, msg)
    print(f"Sent: {subject}")


if __name__ == "__main__":
    main()
