"""
UNIT-TEST ONLY.  Writes files with the exact column layout of Kaggle's train_transaction.csv (394 columns) and
train_identity.csv (41 columns) so the preprocessing / training code can be tested without the licensed data.
The VALUES are random - nothing produced from this file is a research result.
"""
import numpy as np, pandas as pd, os

TX_COLS = (["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD"] + [f"card{i}" for i in range(1, 7)]
           + ["addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain"] + [f"C{i}" for i in range(1, 15)]
           + [f"D{i}" for i in range(1, 16)] + [f"M{i}" for i in range(1, 10)] + [f"V{i}" for i in range(1, 340)])
ID_COLS = ["TransactionID"] + [f"id_{i:02d}" for i in range(1, 39)] + ["DeviceType", "DeviceInfo"]
assert len(TX_COLS) == 394 and len(ID_COLS) == 41


def make_mock(n_tx=60000, out_dir="/tmp/vesta_mock", seed=0, chunk=50000):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    txp, idp = os.path.join(out_dir, "train_transaction.csv"), os.path.join(out_dir, "train_identity.csv")
    for f in (txp, idp):
        if os.path.exists(f):
            os.remove(f)
    n_cards = max(50, n_tx // 7)
    card_first = rng.integers(0, 150, n_cards)
    card_addr = rng.integers(100, 540, (n_cards, 2))
    card_fraud = rng.random(n_cards) < 0.07
    ring_dev = [f"RING-DEV-{k:03d} Build/X" for k in range(40)]
    devs = [f"SM-G{rng.integers(900,999)}F Build/{k}" for k in range(600)] + ["Windows", "iOS Device", "MacOS", "rv:57.0"]
    oses = ["Windows 10", "Windows 7", "iOS 11.1.2", "Android 7.0", "Mac OS X 10_12_6"]; brs = ["chrome 63.0", "safari generic", "firefox 57.0", "edge 16.0"]
    emails = ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com", "icloud.com"]
    tid = 2987000; first_tx = first_id = True
    for start in range(0, n_tx, chunk):
        n = min(chunk, n_tx - start)
        card = np.minimum(rng.zipf(1.35, n), n_cards) - 1
        card = (card * 2654435761 % n_cards).astype(int)
        dt = np.sort(rng.integers(86400, 86400 * 182, n))
        day = dt // 86400
        fr = card_fraud[card] & (rng.random(n) < 0.6)
        d1 = np.maximum(day - card_first[card], 0).astype(float)
        tx = {c: np.full(n, np.nan) for c in TX_COLS}
        tx["TransactionID"] = tid + start + np.arange(n); tx["isFraud"] = fr.astype(int); tx["TransactionDT"] = dt
        tx["TransactionAmt"] = np.round(rng.lognormal(4.2 + 0.5 * fr, 1.0, n), 3)
        tx["ProductCD"] = rng.choice(list("WCHRS"), n, p=[.7, .1, .08, .07, .05])
        tx["card1"] = 1000 + card; tx["card2"] = np.where(rng.random(n) < .02, np.nan, 100 + card % 450); tx["card3"] = 150.0
        tx["card4"] = rng.choice(["visa", "mastercard", "american express", "discover"], n, p=[.65, .3, .03, .02])
        tx["card5"] = 100 + card % 137; tx["card6"] = rng.choice(["debit", "credit"], n, p=[.75, .25])
        tx["addr1"] = np.where(rng.random(n) < .1, np.nan, card_addr[card, rng.integers(0, 2, n)]); tx["addr2"] = np.where(rng.random(n) < .1, np.nan, 87.0)
        tx["dist1"] = np.where(rng.random(n) < .6, np.nan, rng.exponential(60, n)); tx["dist2"] = np.where(rng.random(n) < .93, np.nan, rng.exponential(200, n))
        tx["P_emaildomain"] = np.where(rng.random(n) < .1, None, rng.choice(emails, n)); tx["R_emaildomain"] = np.where(rng.random(n) < .75, None, rng.choice(emails, n))
        for i in range(1, 15):
            tx[f"C{i}"] = np.floor(rng.exponential(1.5 + 4 * fr * (i <= 3), n))
        for i in range(1, 16):
            tx[f"D{i}"] = np.where(rng.random(n) < (0.01 if i == 1 else .4), np.nan, d1 * rng.random(n) if i > 1 else d1)
        for i in range(1, 10):
            tx[f"M{i}"] = np.where(rng.random(n) < .45, None, rng.choice(["T", "F"], n))
        base = rng.normal(0, 1, (n, 12)) + fr[:, None] * 0.6
        for i in range(1, 340):
            v = base[:, i % 12] * (1 + (i % 5)) + rng.normal(0, 1, n)
            tx[f"V{i}"] = np.where(rng.random(n) < (0.5 if i > 50 else 0.0), np.nan, v)
        pd.DataFrame(tx)[TX_COLS].to_csv(txp, mode="a", header=first_tx, index=False); first_tx = False
        has_id = rng.random(n) < 0.24
        k = int(has_id.sum()); idd = {c: np.full(k, np.nan, dtype=object) for c in ID_COLS}
        idd["TransactionID"] = tx["TransactionID"][has_id]
        ringy = fr[has_id] & (rng.random(k) < .7)
        idd["DeviceInfo"] = np.where(ringy, rng.choice(ring_dev, k), rng.choice(devs, k))
        idd["DeviceType"] = rng.choice(["mobile", "desktop"], k); idd["id_30"] = rng.choice(oses, k); idd["id_31"] = rng.choice(brs, k)
        idd["id_33"] = rng.choice(["1920x1080", "1366x768", "2208x1242", "1440x900"], k); idd["id_23"] = rng.choice(["IP_PROXY:TRANSPARENT", "IP_PROXY:ANONYMOUS", None], k, p=[.1, .05, .85])
        for i in range(1, 12):
            idd[f"id_{i:02d}"] = rng.normal(-5, 20, k)
        for c, vals in (("id_12", ["Found", "NotFound"]), ("id_15", ["New", "Found", "Unknown"]), ("id_16", ["Found", "NotFound"]), ("id_27", ["Found"]),
                        ("id_28", ["New", "Found"]), ("id_29", ["Found", "NotFound"]), ("id_34", ["match_status:2", "match_status:1"]),
                        ("id_35", ["T", "F"]), ("id_36", ["T", "F"]), ("id_37", ["T", "F"]), ("id_38", ["T", "F"])):
            idd[c] = rng.choice(vals, k)
        pd.DataFrame(idd)[ID_COLS].to_csv(idp, mode="a", header=first_id, index=False); first_id = False
    return txp, idp


if __name__ == "__main__":
    import sys
    print(make_mock(int(sys.argv[1]) if len(sys.argv) > 1 else 60000))
