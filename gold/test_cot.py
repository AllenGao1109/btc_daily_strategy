"""Gold COT (Commitment of Traders) Managed Money positioning signals.

Disaggregated futures-only report (managed money breakout from 2006).
Gold = CFTC contract market code 088691 (COMEX, Market name contains 'GOLD').
Signal logic (practitioner): Managed Money net long as % of OI, normalized.
  - Contrarian: extreme net long -> sell (overcrowded); extreme net short -> buy.
  - Trend/sentiment: high net long -> stay long (momentum of positioning).
COT released Friday for Tuesday's data -> we lag to the FOLLOWING data availability.
Harness lags weight by 1 more day, so use reported-date alignment (conservative).
"""
import numpy as np, pandas as pd, urllib.request, io, zipfile
from gold.harness import load, backtest, report, SPLIT, metrics

CACHE = "data/processed/gold/cot_gold.csv"

def fetch_cot():
    import os
    if os.path.exists(CACHE):
        d = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        return d
    rows = []
    urls = ['https://www.cftc.gov/files/dea/history/fut_disagg_txt_hist_2006_2016.zip']
    urls += [f'https://www.cftc.gov/files/dea/history/fut_disagg_txt_{yr}.zip' for yr in range(2017, 2027)]
    for u in urls:
        try:
            raw = urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent':'Mozilla/5.0'}), timeout=60).read()
            z = zipfile.ZipFile(io.BytesIO(raw))
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, encoding='latin1')
        except Exception as e:
            print('skip', u, e); continue
        df.columns = [c.strip().strip('"') for c in df.columns]
        name_col = 'Market_and_Exchange_Names'
        code = df['CFTC_Contract_Market_Code'].astype(str).str.strip().str.lstrip('0')
        sub = df[code == '88691']
        dcol = 'Report_Date_as_YYYY-MM-DD' if 'Report_Date_as_YYYY-MM-DD' in df.columns else 'Report_Date_as_MM_DD_YYYY'
        for _, row in sub.iterrows():
            d = pd.to_datetime(str(row[dcol]))
            rows.append({
                'date': d,
                'oi': float(row['Open_Interest_All']),
                'mm_long': float(row['M_Money_Positions_Long_All']),
                'mm_short': float(row['M_Money_Positions_Short_All']),
            })
        print('loaded', u.split('/')[-1], len(sub), 'rows')
    out = pd.DataFrame(rows).drop_duplicates('date').set_index('date').sort_index()
    out['mm_net'] = out['mm_long'] - out['mm_short']
    out['mm_net_pct'] = out['mm_net'] / out['oi']
    out.to_csv(CACHE)
    return out

cot = fetch_cot()
if cot.index.tz is None:
    cot.index = cot.index.tz_localize('UTC')
print(f"COT gold: {cot.index.min().date()} -> {cot.index.max().date()} ({len(cot)} weeks)")
print(cot.tail(3)[['oi','mm_long','mm_short','mm_net','mm_net_pct']])

df = load()
idx = df.index
r = df['close'].pct_change()

# Align weekly COT to daily index. COT 'Report_Date' is Tuesday; published Friday.
# Conservative: shift the signal forward to the Friday after the report date so we
# only use it once public, then ffill. Harness adds 1 more day lag.
# publish ~3 days after report tuesday -> friday
cot_pub = cot.copy()
cot_pub.index = cot_pub.index + pd.Timedelta(days=3)  # Tue data public Fri
sig_raw = cot_pub['mm_net_pct'].reindex(cot_pub.index.union(idx)).sort_index().ffill().reindex(idx)

BH = {}
for k,(a,b) in SPLIT.items():
    m=(idx>=a)&(idx<b); BH[k]=metrics(r,m)
print("\nB&H:", {k:round(BH[k]['sharpe'],2) for k in SPLIT})

def run(weight, label, band=0.10):
    net, held, turn = backtest(df, weight, band=band)
    report(df, net, turn, label)
    sm={}
    for k,(a,b) in SPLIT.items():
        m=(idx>=a)&(idx<b); sm[k]=metrics(net,m)
    beat=(sm['val'] and sm['test'] and sm['val']['sharpe']>BH['val']['sharpe'] and sm['test']['sharpe']>BH['test']['sharpe'])
    print(f"   >>> beats B&H OOS (val&test): {beat}")
    return sm

print("\n" + "="*60 + "\nCOT Managed-Money positioning signals\n" + "="*60)

# z-score of mm_net_pct over trailing window (positioning extreme)
for win in [104, 156, 260]:  # 2,3,5 yr in weeks but we're on daily -> use daily-equiv
    z = (sig_raw - sig_raw.rolling(win*5).mean()) / sig_raw.rolling(win*5).std()
    z = z.clip(-3, 3)

    # CONTRARIAN: high net long -> short. weight = -z scaled
    run((-z*0.5).clip(-1.5,1.5), f"COT contrarian -z (win~{win}w)", band=0.15)

    # SENTIMENT/TREND: high net long -> long. weight = +z (positioning momentum)
    run((z*0.5).clip(-1.5,1.5), f"COT trend +z (win~{win}w)", band=0.15)

# Long-biased contrarian: only reduce when extremely long, never short the bull
z = (sig_raw - sig_raw.rolling(156*5).mean())/sig_raw.rolling(156*5).std()
z = z.clip(-3,3)
w = (1.0 - 0.5*z.clip(lower=0)).clip(0,1.5)  # default long 1.0, trim when crowded long
run(w, "COT long-biased trim-when-crowded (win~3y)", band=0.10)

# Raw level threshold: long when mm_net_pct below its median (not crowded), flat when extreme
med = sig_raw.expanding(min_periods=252).median()
w = pd.Series(1.0, index=idx)
w[sig_raw > sig_raw.expanding(min_periods=252).quantile(0.85)] = 0.0  # stand aside when very crowded
run(w, "COT flat-when-top15pct-crowded long-else", band=0.10)
