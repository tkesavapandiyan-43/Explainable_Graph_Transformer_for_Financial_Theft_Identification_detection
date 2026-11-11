"""
 (1) Environment aggregation - fraud rings share pools of devices / IPs / addresses
 (2) Environment difference  - a stolen personal ID is used by a fraudster whose
                               devices / IPs / addresses differ completely from the
                               victim's own (legit) account under the same ID
 (3) Behavioural anomaly     - late-night bursts, rapid logins, many locations,
                               fast large payments

and injects realistic NOISE so the task is not trivial: shared family devices,
public-WiFi / office IPs, busy delivery addresses, night-owl customers, travellers,
legit sale-day bursts and "camouflaged" fraudsters who mimic normal behaviour.
"""
import numpy as np
import pandas as pd

START = pd.Timestamp("2024-01-01")
DAYS = 240


def _ip(rng, prefix=None):
    if prefix is None:
        prefix = f"{rng.integers(11, 223)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}"
    return f"{prefix}.{rng.integers(1, 254)}"


def _prefix(rng):
    return f"{rng.integers(11, 223)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}"


def _stick(k, rng):
    w = rng.dirichlet(np.ones(k) * 0.9)
    return w / w.sum()


def _legit_times(rng, t0, n, night_owl=False):
    """Daytime-biased activity spread over weeks."""
    span_days = max(DAYS - t0 / 86400.0, 6.0)
    days = np.floor(rng.uniform(0, span_days, size=n))
    if night_owl:
        p = np.full(24, 0.02); p[[22, 23, 0, 1, 2, 3]] = 0.14; p[8:20] = 0.01
    else:
        p = np.full(24, 0.004); p[8:23] = 0.065
    p = p / p.sum()
    hours = rng.choice(24, size=n, p=p)
    return np.sort(t0 + days * 86400 + hours * 3600 + rng.uniform(0, 3600, size=n))


def _burst_times(rng, t0, n, night=True):
    """Concentrated sessions (a few short bursts), mostly at night."""
    n_bursts = int(rng.integers(1, 4))
    span_days = max(DAYS - t0 / 86400.0, 6.0)
    starts = np.floor(rng.uniform(0, span_days, size=n_bursts))
    out = []
    for _ in range(n):
        s = starts[rng.integers(0, n_bursts)]
        if night and rng.random() < 0.75:
            hour = rng.choice([0, 1, 2, 3, 4, 23, 22])
        else:
            hour = rng.integers(8, 22)
        out.append(t0 + s * 86400 + hour * 3600 + rng.exponential(420))
    return np.sort(np.array(out))


def generate_events(n_accounts=6000, fraud_rate=0.17, unlabeled_rate=0.25, seed=7, difficulty=1.0, label_noise=0.03):
    """
    Returns
    -------
    events : DataFrame with columns
             event_id, account_id, person_id, timestamp, event_type, device_id, ip, address, amount, label
             label: 0 normal, 1 fraud, -1 unlabeled (hidden)
    truth  : DataFrame (account_id, true_label, pattern) - ground truth kept separately
    """
    rng = np.random.default_rng(seed)
    office_ips = [_prefix(rng) for _ in range(max(8, n_accounts // 400))]
    busy_addr = [f"ADDR_BIZ_{i:04d}" for i in range(max(10, n_accounts // 300))]
    shared_dev_pool = [f"DEV_SH_{i:04d}" for i in range(max(10, n_accounts // 250))]
    cnt = {"dev": 0, "addr": 0, "acc": 0, "per": 0}

    def new_dev():
        cnt["dev"] += 1
        return f"DEV_{cnt['dev']:06d}"

    def new_addr():
        cnt["addr"] += 1
        return f"ADDR_{cnt['addr']:06d}"

    rows, truth = [], []

    def make_events(acc, per, times, devs, ips, addrs, fraud, amount_scale, pay_frac):
        n = len(times)
        etype = rng.choice(["login", "payment", "order"], size=n, p=[1 - pay_frac - 0.12, pay_frac, 0.12])
        dsel = rng.choice(len(devs), size=n, p=_stick(len(devs), rng))
        isel = rng.choice(len(ips), size=n, p=_stick(len(ips), rng))
        asel = rng.choice(len(addrs), size=n, p=_stick(len(addrs), rng))
        amt = np.where(etype == "login", 0.0,
                       np.round(rng.lognormal(np.log(amount_scale), 0.45 if fraud else 0.7, size=n), 2))
        for k in range(n):
            rows.append((acc, per, times[k], etype[k], devs[dsel[k]], ips[isel[k]], addrs[asel[k]], amt[k]))

    ring_pending = []          # (acc, pid, t0) of accounts that will be assigned to a fraud ring
    n_target = n_accounts

    while cnt["acc"] < n_target:
        cnt["per"] += 1
        pid = f"P_{cnt['per']:06d}"
        u = rng.random()
        n_sib = 1 if u < 0.62 else (2 if u < 0.88 else (3 if u < 0.97 else 4))
        n_sib = min(n_sib, n_target - cnt["acc"])
        home_prefix = _prefix(rng)
        own_devs = [new_dev() for _ in range(1 + int(rng.random() < 0.45) + int(rng.random() < 0.12))]
        own_addr = [new_addr() for _ in range(1 + int(rng.random() < 0.35))]
        mobile_ips = [_ip(rng) for _ in range(1 + int(rng.random() < 0.6))]
        fraud_slot = int(rng.integers(0, n_sib)) if rng.random() < fraud_rate * 1.05 else -1

        for s in range(n_sib):
            cnt["acc"] += 1
            acc = f"ACC_{cnt['acc']:06d}"
            t0 = rng.uniform(0, DAYS * 0.65) * 86400
            if s == fraud_slot:
                pattern = str(rng.choice(["ring", "diff", "behav", "camo"], p=[0.36, 0.24, 0.18, 0.22]))
                if pattern == "ring":
                    ring_pending.append((acc, pid, t0, own_devs, own_addr))
                    truth.append((acc, 1, "ring"))
                    continue
                if pattern == "diff":
                    devs = [new_dev() for _ in range(int(rng.integers(1, 4)))]
                    ips = [_ip(rng) for _ in range(int(rng.integers(2, 6)))]
                    addrs = [new_addr() for _ in range(int(rng.integers(1, 3)))]
                    times = _burst_times(rng, t0, int(rng.poisson(18)) + 5, night=rng.random() < 0.45) if rng.random() < 0.75 else _legit_times(rng, t0, int(rng.poisson(14)) + 5)
                    make_events(acc, pid, times, devs, ips, addrs, True, 260, 0.35)
                elif pattern == "behav":
                    devs = own_devs[:1] + [new_dev()]
                    ips = mobile_ips + [_ip(rng) for _ in range(int(rng.integers(2, 6)))]
                    addrs = own_addr + [new_addr()]
                    times = _burst_times(rng, t0, int(rng.poisson(24)) + 8, night=True)
                    make_events(acc, pid, times, devs, ips, addrs, True, 300, 0.4)
                else:   # camouflaged
                    devs = [new_dev()] + (own_devs[:1] if rng.random() < 0.6 else [])
                    ips = [_ip(rng, home_prefix)] if rng.random() < 0.6 else [_ip(rng)]
                    addrs = [new_addr()] if rng.random() < 0.7 else list(own_addr)
                    n_ev = int(rng.poisson(16)) + 5
                    times = _legit_times(rng, t0, n_ev, night_owl=rng.random() < 0.25)
                    if rng.random() < 0.35:
                        k = max(2, n_ev // 5)
                        times[:k] = _burst_times(rng, t0, k, night=rng.random() < 0.5)
                        times = np.sort(times)
                    make_events(acc, pid, times, devs, ips, addrs, False, 140, 0.35)
                truth.append((acc, 1, pattern))
            else:
                night_owl = rng.random() < 0.17 * difficulty
                traveller = rng.random() < 0.17 * difficulty
                devs = list(own_devs)
                if rng.random() < 0.16 * difficulty:
                    devs.append(str(rng.choice(shared_dev_pool)))
                if s > 0 and rng.random() < 0.30:
                    devs = [new_dev()]
                ips = [_ip(rng, home_prefix)] + list(mobile_ips[: int(rng.integers(1, 3))])
                if rng.random() < 0.22 * difficulty:
                    ips.append(_ip(rng, office_ips[rng.integers(0, len(office_ips))]))
                if traveller:
                    ips += [_ip(rng) for _ in range(int(rng.integers(2, 5)))]
                addrs = list(own_addr)
                if rng.random() < 0.18 * difficulty:
                    addrs.append(busy_addr[rng.integers(0, len(busy_addr))])
                n_ev = int(rng.poisson(20)) + 4
                times = _legit_times(rng, t0, n_ev, night_owl)
                if rng.random() < 0.11 * difficulty:
                    k = max(3, n_ev // 4)
                    times[:k] = _burst_times(rng, t0, k, night=False)
                    times = np.sort(times)
                make_events(acc, pid, times, devs, ips, addrs, False, 110 + rng.random() * 90, 0.38)
                truth.append((acc, 0, "normal"))

    # ---- fraud rings: members share device / IP-prefix / address pools --------------------
    i = 0
    while i < len(ring_pending):
        size = int(rng.integers(3, 8))
        grp = ring_pending[i:i + size]
        i += size
        pool_dev = [new_dev() for _ in range(max(1, len(grp) // 2))]
        pool_pref = [_prefix(rng) for _ in range(max(1, len(grp) // 3))]
        pool_addr = [new_addr() for _ in range(max(1, len(grp) // 3))]
        for (acc, pid, t0, own_devs, own_addr) in grp:
            devs = list(rng.choice(pool_dev, size=min(len(pool_dev), int(rng.integers(1, 3))), replace=False))
            if rng.random() < 0.25:           # sloppy ring member also uses a fresh device
                devs.append(new_dev())
            ips = [_ip(rng, pool_pref[rng.integers(0, len(pool_pref))]) for _ in range(int(rng.integers(1, 4)))]
            addrs = list(rng.choice(pool_addr, size=min(len(pool_addr), int(rng.integers(1, 3))), replace=False))
            night = rng.random() < 0.65
            times = _burst_times(rng, t0, int(rng.poisson(18)) + 6, night=night) if rng.random() < 0.68 \
                else _legit_times(rng, t0, int(rng.poisson(14)) + 5)
            make_events(acc, pid, times, devs, ips, addrs, True, 280, 0.35)

    ev = pd.DataFrame(rows, columns=["account_id", "person_id", "ts", "event_type", "device_id", "ip", "address", "amount"])
    ev["timestamp"] = START + pd.to_timedelta(ev["ts"].clip(0, DAYS * 86400 - 1), unit="s")
    ev = ev.drop(columns="ts").sort_values("timestamp").reset_index(drop=True)
    ev.insert(0, "event_id", [f"EV_{i:07d}" for i in range(len(ev))])
    truth = pd.DataFrame(truth, columns=["account_id", "true_label", "pattern"])

    # labels: part of the accounts are unlabeled (-1) -> semi-supervised setting as in the paper
    lab = truth.set_index("account_id")["true_label"].copy()
    flip = rng.random(len(lab)) < label_noise          # imperfect manual labels
    lab[flip] = 1 - lab[flip]
    hide = rng.random(len(lab)) < unlabeled_rate
    lab[hide] = -1
    ev["label"] = ev["account_id"].map(lab).astype(int)
    ev["timestamp"] = ev["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    ev["amount"] = ev["amount"].astype(float)
    return ev[["event_id", "account_id", "person_id", "timestamp", "event_type",
               "device_id", "ip", "address", "amount", "label"]], truth


if __name__ == "__main__":
    e, t = generate_events(500, seed=1)
    print(e.head(), len(e), t.true_label.mean())
