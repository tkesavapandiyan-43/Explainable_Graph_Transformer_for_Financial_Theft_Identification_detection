"""
Real-data pipeline for the Kaggle IEEE-CIS Fraud Detection (Vesta) files  train_transaction.csv / train_identity.csv.

Design decisions (all documented in workflow.txt)
-------------------------------------------------
Graph entities (nodes)
  Account   = card1|card2|card3|card5|addr1| (floor(TransactionDT/86400) - D1)   the paper's "bank-card, address and
              days-related fields" account construction  (D1 = days since the card's first transaction)
  Identity  = card1|...|card6 (the bank-card as "verified personal ID"; accounts of one card are siblings)
  Device    = DeviceInfo | id_30 (OS) | id_31 (browser) | id_33 (screen)          - a device fingerprint
  Network   = P_emaildomain | R_emaildomain | id_23 (proxy type)   Vesta has NO raw IP address; these are the only
              network-level signals in the public files (the paper also uses "network information")
  Address   = addr1|addr2 (billing region / country codes)
  OS and browser are NOT separate nodes: with a handful of distinct values each they would connect tens of thousands
  of unrelated accounts and carry almost no evidence. They are device-fingerprint components and frequency features.
  Transactions are NOT nodes: they are the events of the account (behaviour sequence + edge counts).

Event features (x_*)  - every one is derived from the raw columns; every statistic is fitted on TRAIN accounts only
  amount (log, cents), ProductCD, card4/card6, dist1/2, C1-C14 (log1p), D1-D15, M1-M9, V1-V339 -> 24 PCA components,
  missing-value rates, e-mail domain frequency + match, identity numerics id_01-id_11, categorical identity fields
  (frequency-encoded), screen size, has-identity flag.

Everything is read in chunks (default 100 000 rows); the full dataset is processed - nothing is sub-sampled.
"""
import os, pickle, time
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA

REF_DATE = pd.Timestamp("2017-12-01")          # Kaggle gives TransactionDT as an offset; this reference date is the
                                               # community-accepted assumption (only used to display dates)
KEY_COLS = ["TransactionID", "TransactionDT", "isFraud", "card1", "card2", "card3", "card4", "card5", "card6",
            "addr1", "addr2", "D1", "P_emaildomain", "R_emaildomain"]
C_COLS = [f"C{i}" for i in range(1, 15)]
D_COLS = [f"D{i}" for i in range(1, 16)]
M_COLS = [f"M{i}" for i in range(1, 10)]
V_COLS = [f"V{i}" for i in range(1, 340)]
ID_NUM = [f"id_{i:02d}" for i in range(1, 12)]
ID_CAT = ["DeviceType", "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29", "id_30", "id_31",
          "id_34", "id_35", "id_36", "id_37", "id_38"]
ID_COLS = ID_NUM + ID_CAT + ["id_33", "DeviceInfo"]
TX_USE = set(KEY_COLS + C_COLS + D_COLS + M_COLS + V_COLS + ["TransactionAmt", "ProductCD", "dist1", "dist2"])
NUM_SCALAR = ["amt_log", "dist1_log", "dist2_log"] + C_COLS + D_COLS + ID_NUM + ["screen_log"]
NUM_ALL = NUM_SCALAR + V_COLS
PRODUCTS = ["W", "C", "H", "R", "S"]
CARD4 = ["visa", "mastercard", "american express", "discover"]
CARD6 = ["debit", "credit"]
FREQ_COLS = ["P_emaildomain", "R_emaildomain"] + ID_CAT


def _canon(cols):
    return [str(c).replace("-", "_") for c in cols]


def _read_identity(path):
    if not path:
        return None
    df = pd.read_csv(path, low_memory=False)
    df.columns = _canon(df.columns)
    keep = ["TransactionID"] + [c for c in ID_COLS if c in df.columns]
    df = df[keep].drop_duplicates("TransactionID").set_index("TransactionID")
    return df


def _iter_chunks(tx_csv, id_df, usecols=None, chunksize=100_000):
    start = 0
    for ch in pd.read_csv(tx_csv, usecols=usecols or (lambda c: c in TX_USE), chunksize=chunksize, low_memory=False):
        ch.columns = _canon(ch.columns)
        if id_df is not None:
            ch = ch.join(id_df, on="TransactionID")
        yield start, ch.reset_index(drop=True)
        start += len(ch)


def _str(df, col):
    return df[col].fillna("").astype(str) if col in df.columns else pd.Series([""] * len(df), index=df.index)


def entity_strings(df):
    """Device / network / address strings for a chunk (shared by the report, the fit and the transform)."""
    dev_parts = [_str(df, c) for c in ("DeviceInfo", "id_30", "id_31", "id_33")]
    has = (dev_parts[0] != "") | (dev_parts[1] != "") | (dev_parts[2] != "")
    device = (dev_parts[0] + "|" + dev_parts[1] + "|" + dev_parts[2] + "|" + dev_parts[3]).where(has, "")
    pe, re_, px = _str(df, "P_emaildomain"), _str(df, "R_emaildomain"), _str(df, "id_23")
    network = (pe + "|" + re_ + "|" + px).where((pe != "") | (re_ != ""), "")
    a1 = df["addr1"].round().astype("Int64").astype(str) if "addr1" in df.columns else pd.Series([""] * len(df))
    a2 = df["addr2"].round().astype("Int64").astype(str) if "addr2" in df.columns else pd.Series([""] * len(df))
    address = (a1 + "|" + a2).where(df["addr1"].notna(), "") if "addr1" in df.columns else a1
    return device, network, address


def _account_keys(keys):
    """account / person codes from the key columns of ALL rows."""
    first_day = np.floor(keys["TransactionDT"] / 86400.0 - keys["D1"])
    acct = keys.assign(_fd=first_day).groupby(["card1", "card2", "card3", "card5", "addr1", "_fd"], dropna=False, sort=False, observed=True).ngroup()
    person = keys.groupby(["card1", "card2", "card3", "card4", "card5", "card6"], dropna=False, sort=False, observed=True).ngroup()
    return acct.to_numpy(), person.to_numpy()


def _read_keys(tx_csv, chunksize):
    parts = []
    for ch in pd.read_csv(tx_csv, usecols=lambda c: c.replace("-", "_") in set(KEY_COLS), chunksize=chunksize, low_memory=False):
        ch.columns = _canon(ch.columns)
        parts.append(ch)
    keys = pd.concat(parts, ignore_index=True)
    if "isFraud" not in keys.columns:
        keys["isFraud"] = -1
    for c in ("card4", "card6", "P_emaildomain", "R_emaildomain"):
        if c in keys.columns:
            keys[c] = keys[c].astype("category")
    return keys


class VestaPreprocessor:
    """Fit once on the training accounts of the real data, then transform any compatible file."""

    def __init__(self):
        self.state = None

    # ------------------------------------------------------------------ fit
    def fit_transform(self, tx_csv, id_csv=None, split="time", seed=42, n_pca=24, chunksize=100_000, log=print):
        t0 = time.time()
        log(f"[1/4] reading keys of {os.path.basename(tx_csv)} ...")
        keys = _read_keys(tx_csv, chunksize)
        acct, person = _account_keys(keys)
        n_acc = acct.max() + 1
        lab_acc = pd.Series(keys["isFraud"].to_numpy()).groupby(acct).max().reindex(range(n_acc)).to_numpy()
        first_seen = pd.Series(keys["TransactionDT"].to_numpy()).groupby(acct).min().reindex(range(n_acc)).to_numpy()
        acc_split = self._split(lab_acc, first_seen, split, seed)
        row_split = acc_split[acct]
        log(f"      {len(keys):,} transactions -> {n_acc:,} accounts, {person.max()+1:,} personal IDs; "
            f"split={split}: train {int((acc_split==0).sum()):,} / val {int((acc_split==1).sum()):,} / test {int((acc_split==2).sum()):,} accounts")
        id_df = _read_identity(id_csv)
        # ---- pass 2: statistics on TRAIN rows only
        log("[2/4] fitting normalisation statistics on TRAIN accounts (pass over the file) ...")
        s1 = np.zeros(len(NUM_ALL)); s2 = np.zeros(len(NUM_ALL)); cn = np.zeros(len(NUM_ALL))
        counts = {c: {} for c in FREQ_COLS}; n_train = 0
        for st, ch in _iter_chunks(tx_csv, id_df, chunksize=chunksize):
            m = row_split[st:st + len(ch)] == 0
            if not m.any():
                continue
            raw = self._raw_numeric(ch[m])
            v = raw.to_numpy(np.float64)
            ok = ~np.isnan(v)
            s1 += np.where(ok, v, 0).sum(0); s2 += (np.where(ok, v, 0) ** 2).sum(0); cn += ok.sum(0)
            n_train += int(m.sum())
            for c in FREQ_COLS:
                if c in ch.columns:
                    for k, n in ch.loc[m, c].value_counts().items():
                        counts[c][k] = counts[c].get(k, 0) + int(n)
        mean = s1 / np.maximum(cn, 1)
        std = np.sqrt(np.maximum(s2 / np.maximum(cn, 1) - mean ** 2, 0)); std[std < 1e-6] = 1.0
        # ---- pass 3: PCA of the V block
        log(f"[3/4] fitting {n_pca}-component PCA on the {len(V_COLS)} V features (train accounts only) ...")
        vi = [NUM_ALL.index(c) for c in V_COLS]
        pca = IncrementalPCA(n_components=n_pca)
        self.state = dict(mean=mean, std=std, counts=counts, n_train=n_train, pca=pca, split_mode=split, seed=seed,
                          n_pca=n_pca, vi=vi)
        for st, ch in _iter_chunks(tx_csv, id_df, usecols=lambda c: c.replace("-", "_") in set(V_COLS + ["TransactionID"]), chunksize=chunksize):
            m = row_split[st:st + len(ch)] == 0
            if m.sum() >= n_pca:
                z = self._z(self._raw_numeric_v(ch[m]), vi)
                pca.partial_fit(z)
        self.state["pca_explained"] = float(pca.explained_variance_ratio_.sum())
        log(f"      PCA keeps {self.state['pca_explained']:.1%} of the V-feature variance")
        # ---- pass 4: transform every row
        log("[4/4] building event features for ALL transactions ...")
        parts = []
        for st, ch in _iter_chunks(tx_csv, id_df, chunksize=chunksize):
            sl = slice(st, st + len(ch))
            parts.append(self._events(ch, acct[sl], person[sl], lab_acc[acct[sl]], row_split[sl]))
            log(f"      {min(st+len(ch), len(keys)):,}/{len(keys):,}")
        ev = pd.concat(parts, ignore_index=True)
        self.state["x_cols"] = [c for c in ev.columns if c.startswith("x_")]
        info = dict(n_transactions=len(keys), n_accounts=int(n_acc), n_persons=int(person.max() + 1),
                    fraud_accounts=int((lab_acc == 1).sum()), legit_accounts=int((lab_acc == 0).sum()),
                    fraud_transactions=int((keys["isFraud"] == 1).sum()), split_mode=split,
                    split_accounts=[int((acc_split == k).sum()) for k in range(3)],
                    n_event_features=len(self.state["x_cols"]), seconds=time.time() - t0)
        log(f"done in {info['seconds']:.0f}s  ({len(self.state['x_cols'])} event features per transaction)")
        return ev, info

    # ------------------------------------------------------------------ transform of a new file
    def transform_files(self, tx_csv, id_csv=None, chunksize=100_000):
        keys = _read_keys(tx_csv, chunksize)
        acct, person = _account_keys(keys)
        n_acc = acct.max() + 1
        lab = pd.Series(keys["isFraud"].to_numpy()).groupby(acct).max().reindex(range(n_acc)).to_numpy()
        id_df = _read_identity(id_csv)
        parts = []
        for st, ch in _iter_chunks(tx_csv, id_df, chunksize=chunksize):
            sl = slice(st, st + len(ch))
            parts.append(self._events(ch, acct[sl], person[sl], lab[acct[sl]], np.full(len(ch), -1)))
        return pd.concat(parts, ignore_index=True)

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _split(label, first_seen, mode, seed, frac=(0.6, 0.2, 0.2)):
        n = len(label); out = np.zeros(n, int)
        if mode == "time":                                   # oldest accounts train, newest accounts test
            order = np.lexsort((np.arange(n), first_seen))
            a, b = int(n * frac[0]), int(n * (frac[0] + frac[1]))
            out[order[a:b]] = 1; out[order[b:]] = 2
        elif mode == "random":                               # paper's stratified random 3:1:1
            from .train import make_split
            tr, va, te = make_split(np.where(np.isnan(label), -1, label).astype(int), seed)
            out[:] = -1; out[tr] = 0; out[va] = 1; out[te] = 2
        else:
            raise ValueError("split must be 'time' or 'random'")
        return out

    @staticmethod
    def _raw_numeric_v(ch):
        return ch[[c for c in V_COLS if c in ch.columns]].reindex(columns=V_COLS).astype(np.float32)

    @staticmethod
    def _raw_numeric(ch):
        n = len(ch)
        nanv = lambda: np.full(n, np.nan, np.float32)
        col = lambda c: ch[c].to_numpy(np.float32) if c in ch.columns else nanv()
        d = {"amt_log": np.log1p(np.maximum(col("TransactionAmt"), 0)), "dist1_log": np.log1p(np.maximum(col("dist1"), 0)),
             "dist2_log": np.log1p(np.maximum(col("dist2"), 0))}
        for c in C_COLS:
            d[c] = np.log1p(np.maximum(col(c), 0))
        for c in D_COLS:
            d[c] = col(c)
        for c in ID_NUM:
            d[c] = col(c)
        if "id_33" in ch.columns:
            wh = ch["id_33"].astype(str).str.extract(r"(\d+)x(\d+)").astype(float)
            d["screen_log"] = np.log1p((wh[0] * wh[1]).to_numpy(np.float32))
        else:
            d["screen_log"] = nanv()
        out = pd.DataFrame(d, columns=NUM_SCALAR)
        return pd.concat([out, VestaPreprocessor._raw_numeric_v(ch).reset_index(drop=True)], axis=1)[NUM_ALL]

    def _z(self, raw, idx=None):
        st = self.state
        v = raw.to_numpy(np.float32)
        mu = st["mean"][idx] if idx is not None else st["mean"]
        sd = st["std"][idx] if idx is not None else st["std"]
        return np.nan_to_num(np.clip((v - mu) / sd, -6, 6), nan=0.0).astype(np.float32)

    def _freq(self, series, col):
        cnt = self.state["counts"][col]; nt = self.state["n_train"]
        return series.map(lambda k: np.log1p(cnt.get(k, 0)) / np.log1p(nt) if pd.notna(k) else 0.0).to_numpy(np.float32)

    def _events(self, ch, acct, person, label, split):
        st = self.state
        n = len(ch)
        raw = self._raw_numeric(ch)
        z = self._z(raw)
        zs = pd.DataFrame(z[:, :len(NUM_SCALAR)], columns=["x_" + c for c in NUM_SCALAR])
        vpca = st["pca"].transform(z[:, st["vi"]]).astype(np.float32)
        vpca = vpca / np.sqrt(st["pca"].explained_variance_[None, :] + 1e-6)          # unit variance components
        X = {c: zs[c].to_numpy() for c in zs.columns}
        amt = ch["TransactionAmt"].to_numpy(np.float32)
        X["x_amt_cents"] = (amt - np.floor(amt)).astype(np.float32)
        for p in PRODUCTS:
            X[f"x_prod_{p}"] = (ch["ProductCD"] == p).to_numpy(np.float32)
        for c in CARD4:
            X[f"x_card4_{c.split()[0]}"] = (ch["card4"] == c).to_numpy(np.float32)
        for c in CARD6:
            X[f"x_card6_{c}"] = (ch["card6"] == c).to_numpy(np.float32)
        mmat = np.stack([ch[c].map({"T": 1.0, "F": -1.0}).fillna(0.0).to_numpy(np.float32) if c in ch.columns else np.zeros(n, np.float32) for c in M_COLS], 1)
        for k, c in enumerate(M_COLS):
            X[f"x_{c}"] = mmat[:, k]
        for k in range(vpca.shape[1]):
            X[f"x_Vpca{k:02d}"] = vpca[:, k]
        X["x_miss_D"] = raw[D_COLS].isna().mean(1).to_numpy(np.float32)
        X["x_miss_V"] = raw[V_COLS].isna().mean(1).to_numpy(np.float32)
        X["x_miss_M"] = np.mean([ch[c].isna().to_numpy() for c in M_COLS if c in ch.columns] or [np.zeros(n)], 0).astype(np.float32)
        X["x_email_match"] = ((ch["P_emaildomain"] == ch["R_emaildomain"]) & ch["P_emaildomain"].notna()).to_numpy(np.float32)
        X["x_has_identity"] = ch[[c for c in ID_NUM + ["DeviceType"] if c in ch.columns]].notna().any(axis=1).to_numpy(np.float32) if any(c in ch.columns for c in ID_NUM) else np.zeros(n, np.float32)
        for c in FREQ_COLS:
            X["x_freq_" + c] = self._freq(ch[c], c) if c in ch.columns else np.zeros(n, np.float32)
        dev, net, adr = entity_strings(ch)
        base = pd.DataFrame({
            "event_id": ch["TransactionID"].to_numpy(), "account_id": ["A_%d" % a for a in acct],
            "person_id": ["P_%d" % p for p in person],
            "timestamp": REF_DATE + pd.to_timedelta(ch["TransactionDT"].to_numpy(), unit="s"),
            "event_type": "payment", "device_id": dev.to_numpy(), "ip": net.to_numpy(), "address": adr.to_numpy(),
            "amount": amt, "label": np.where(np.isnan(label), -1, label).astype(int), "split": split})
        return pd.concat([base, pd.DataFrame(X)], axis=1)

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.state, f)

    @classmethod
    def load(cls, path):
        o = cls()
        with open(path, "rb") as f:
            o.state = pickle.load(f)
        return o


# ======================================================================================= dataset report
def dataset_report(tx_csv, id_csv=None, out_path="outputs/dataset_report.txt", chunksize=100_000, log=print):
    """Full audit of the raw files BEFORE training: rows, columns, labels, missing values, duplicates, uniques, dates."""
    L = []
    P = L.append
    t0 = time.time()
    P("DATASET REPORT  (generated by egt.vesta.dataset_report from the real files - nothing estimated)")
    P("=" * 100)
    id_df_full = None
    # ---------------- transaction file
    rows = 0; miss = None; id_arrays = []; hashes = []; cols = None; dtypes = None
    fraud = 0; dts = []; ent = {"device": set(), "network": set(), "address": set()}
    os_, br, dtype_, dinfo = set(), set(), set(), set()
    first = True
    for ch in pd.read_csv(tx_csv, chunksize=chunksize, low_memory=False):
        ch.columns = _canon(ch.columns)
        if first:
            cols = list(ch.columns); dtypes = ch.dtypes.astype(str).value_counts().to_dict(); miss = np.zeros(len(cols), np.int64); first = False
        rows += len(ch); miss += ch[cols].isna().sum().to_numpy()
        if "isFraud" in ch.columns:
            fraud += int((ch["isFraud"] == 1).sum())
        hashes.append(pd.util.hash_pandas_object(ch.drop(columns=["TransactionID"], errors="ignore"), index=False).to_numpy())
        id_arrays.append(ch["TransactionID"].to_numpy())
        dts.append((ch["TransactionDT"].min(), ch["TransactionDT"].max()))
    h = np.concatenate(hashes); n_dup_rows = int(len(h) - len(np.unique(h)))
    ids_all = np.concatenate(id_arrays); n_dup_id = int(len(ids_all) - len(np.unique(ids_all)))
    P(f"\nFILE 1: {os.path.basename(tx_csv)}")
    P(f"  rows (transactions)        : {rows:,}")
    P(f"  columns                    : {len(cols)}   (dtypes: {dtypes})")
    P(f"  fraud transactions         : {fraud:,}  ({fraud/rows:.3%})   legitimate: {rows-fraud:,}")
    P(f"  duplicated TransactionID   : {n_dup_id:,}   duplicated full rows (excluding the ID): {n_dup_rows:,}")
    tot = rows * len(cols)
    P(f"  missing cells              : {int(miss.sum()):,} of {tot:,} ({miss.sum()/tot:.2%})")
    order = np.argsort(-miss)[:25]
    P("  columns with most missing values:")
    for j in order:
        P(f"      {cols[j]:<16s} {miss[j]:>9,}  ({miss[j]/rows:.1%})")
    lo, hi = min(a for a, _ in dts), max(b for _, b in dts)
    P(f"  TransactionDT range        : {lo:,} .. {hi:,} s  = {(hi-lo)/86400:.1f} days  "
      f"(displayed as {REF_DATE.date()} .. {(REF_DATE+pd.to_timedelta(hi,'s')).date()} assuming reference {REF_DATE.date()})")
    groups = {"transaction": ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "isFraud"],
              "card (6)": [c for c in cols if c.startswith("card")], "address / distance": ["addr1", "addr2", "dist1", "dist2"],
              "email domains": ["P_emaildomain", "R_emaildomain"], "C counting features": [c for c in cols if c[0] == "C" and c[1:].isdigit()],
              "D time-delta features": [c for c in cols if c[0] == "D" and c[1:].isdigit()], "M match flags": [c for c in cols if c[0] == "M" and c[1:].isdigit()],
              "V engineered features": [c for c in cols if c[0] == "V" and c[1:].isdigit()]}
    P("  column groups: " + "; ".join(f"{k}={sum(1 for c in v if c in cols)}" for k, v in groups.items()))
    # ---------------- identity file
    id_df = None
    if id_csv:
        raw = pd.read_csv(id_csv, low_memory=False); raw.columns = _canon(raw.columns)
        n_id_dup = int(raw["TransactionID"].duplicated().sum())
        P(f"\nFILE 2: {os.path.basename(id_csv)}")
        P(f"  rows                       : {len(raw):,}   ({len(raw)/rows:.1%} of transactions have identity information)")
        P(f"  columns                    : {raw.shape[1]}")
        P(f"  duplicated TransactionID   : {n_id_dup:,}   full duplicate rows: {int(raw.duplicated().sum()):,}")
        P(f"  missing cells              : {int(raw.isna().sum().sum()):,} of {raw.size:,} ({raw.isna().sum().sum()/raw.size:.2%})")
        for c, name in (("id_30", "operating systems (OS)"), ("id_31", "browsers"), ("DeviceType", "device types"),
                        ("DeviceInfo", "DeviceInfo values"), ("id_33", "screen resolutions"), ("id_23", "proxy types")):
            if c in raw.columns:
                P(f"  unique {name:<24s}: {raw[c].nunique():,}   top: {raw[c].value_counts().head(4).to_dict()}")
        id_df = _read_identity(id_csv)
    # ---------------- entities and accounts (same construction as training)
    keys = _read_keys(tx_csv, chunksize)
    acct, person = _account_keys(keys)
    n_acc = acct.max() + 1
    lab = pd.Series(keys["isFraud"].to_numpy()).groupby(acct).max().reindex(range(n_acc))
    for st, ch in _iter_chunks(tx_csv, id_df, chunksize=chunksize):
        d, n, a = entity_strings(ch)
        ent["device"].update(d[d != ""].unique()); ent["network"].update(n[n != ""].unique()); ent["address"].update(a[a != ""].unique())
    P("\nGRAPH ENTITIES BUILT FROM THE FILES (same construction as used for training)")
    P(f"  accounts (card1|card2|card3|card5|addr1|first-seen-day) : {n_acc:,}")
    P(f"    accounts with fraud label 1 / 0                        : {int((lab==1).sum()):,} / {int((lab==0).sum()):,}  (account is fraud if any of its transactions is)")
    P(f"    transactions per account  mean {rows/n_acc:.2f}, median {int(pd.Series(acct).value_counts().median())}, max {int(pd.Series(acct).value_counts().max()):,}; single-transaction accounts: {int((pd.Series(acct).value_counts()==1).sum()):,}")
    P(f"  personal identities (card1..card6 combinations)          : {person.max()+1:,}")
    P(f"  unique devices (DeviceInfo|OS|browser|screen fingerprint): {len(ent['device']):,}")
    P(f"  unique networks (P_email|R_email|proxy-type; no raw IP in Vesta): {len(ent['network']):,}")
    P(f"  unique addresses (addr1|addr2)                           : {len(ent['address']):,}")
    P(f"  unlabeled accounts                                       : {int(lab.isna().sum()):,}")
    P(f"\nreport generated in {time.time()-t0:.0f}s")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    open(out_path, "w").write("\n".join(L))
    log("\n".join(L))
    return "\n".join(L)
