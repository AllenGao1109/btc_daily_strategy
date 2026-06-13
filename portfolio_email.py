"""Daily multi-asset (Black-Litterman) portfolio email: target weights + action + chart.

Reuses the SMTP infra from daily_email.py (git-ignored email_config.json). --dry-run
previews without sending.
"""
from __future__ import annotations
import argparse, traceback
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from daily_email import load_email_config, send_email
from portfolio.report import build_report, render_html, make_chart, NAMES_ZH


def compose():
    rep = build_report(force=True)
    Path("logs").mkdir(exist_ok=True)
    chart = make_chart(rep, f"logs/portfolio_equity_{rep['as_of']}.png")
    is_trade = rep["action"] == "REBALANCE"
    tw = rep["target"]
    top = " ".join(f"{a.upper()}{tw[a]*100:.0f}%" for a in rep["assets"] if tw[a] > 0.01)
    subject = (f"[组合 {rep['as_of']}] {'⚠️ 调仓 REBALANCE' if is_trade else '持有 HOLD'} · "
               f"{top} 现金{rep['cash_target']*100:.0f}%")
    text = (f"多资产组合 {rep['as_of']} - {rep['action']} (drift {rep['drift']})\n"
            + "目标权重: " + ", ".join(f"{a.upper()} {tw[a]*100:.0f}%" for a in rep['assets'])
            + f", 现金 {rep['cash_target']*100:.0f}%\n"
            + "观点: " + ", ".join(f"{a.upper()} {rep['views'][a]:+.2f}" for a in rep['assets'])
            + f"\n组合 Sharpe {rep['stats']['full_sharpe']} / 回撤 {rep['stats']['full_dd']}% "
            + f"vs 等权 {rep['stats']['ew_sharpe']} / {rep['stats']['ew_dd']}%")
    html = render_html(rep)
    return subject, text, html, chart


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    try:
        subject, text, html, chart = compose()
    except Exception:
        subject = "[组合] ERROR"; text = traceback.format_exc(); html = f"<pre>{text}</pre>"; chart = None
    if a.dry_run:
        print("SUBJECT:", subject); print(text)
        Path("logs").mkdir(exist_ok=True); Path("logs/portfolio_email_preview.html").write_text(html)
        print("preview -> logs/portfolio_email_preview.html"); return
    ecfg = load_email_config()
    root = MIMEMultipart("related"); root["Subject"] = subject
    root["From"] = ecfg["sender"]; root["To"] = ecfg["recipient"]
    alt = MIMEMultipart("alternative"); alt.attach(MIMEText(text, "plain", "utf-8"))
    alt.attach(MIMEText(html, "html", "utf-8")); root.attach(alt)
    if chart:
        with open(chart, "rb") as fh: img = MIMEImage(fh.read())
        img.add_header("Content-ID", "<portchart>"); img.add_header("Content-Disposition", "inline")
        root.attach(img)
    send_email(ecfg, root); print(f"Sent: {subject}")


if __name__ == "__main__":
    main()
