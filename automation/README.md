# Daily position-guidance email (macOS launchd)

Sends you a daily email with today's BTC `factor_composite` position guidance:
the model's signal target, your band's current holding, and whether a position
change (REBALANCE) is signaled today — plus the full multi-band table.

## 1. Set up email credentials (one time)

Copy the template and fill in your real values:

```bash
cd /Users/gaozhiyuan/Desktop/btc_daily_strategy
cp email_config.example.json email_config.json
# then edit email_config.json
```

`email_config.json` is git-ignored — your password never gets committed.

**Gmail:** set `smtp_host` to `smtp.gmail.com`, port `587`, and use a 16-char
**App Password** (Google Account → Security → 2-Step Verification → App passwords),
NOT your normal password. `username` and `sender` are your Gmail address;
`recipient` is wherever you want the email (can be the same address).

Other providers: use their SMTP host/port (STARTTLS).

## 2. Preview without sending

```bash
PYTHONPATH=. python3 daily_email.py --dry-run        # fetches latest + prints
```

## 3. Send a real test email now

```bash
PYTHONPATH=. python3 daily_email.py
```

## 4. Install the daily schedule (runs at 09:00 local)

```bash
cp automation/com.btcstrategy.dailyemail.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.btcstrategy.dailyemail.plist
# verify it is registered:
launchctl list | grep btcstrategy
```

Change the run time by editing `Hour`/`Minute` in the plist, then reload:

```bash
launchctl unload ~/Library/LaunchAgents/com.btcstrategy.dailyemail.plist
cp automation/com.btcstrategy.dailyemail.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.btcstrategy.dailyemail.plist
```

## Notes

- The job refreshes the latest daily bar before composing the email.
- Logs: `logs/daily_email.log` and `logs/daily_email.err`.
- The Mac must be awake at the scheduled time; if asleep, launchd runs the job
  at the next wake. For a guaranteed run, keep the Mac awake or pick a time it is.
- This is research guidance you act on manually — it does not place orders.
- To stop: `launchctl unload ~/Library/LaunchAgents/com.btcstrategy.dailyemail.plist`
