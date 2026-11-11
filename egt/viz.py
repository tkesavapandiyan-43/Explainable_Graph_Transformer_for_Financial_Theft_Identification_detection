"""
Standardised transaction-graph visualisation (always the same visual language, dynamic content).

  node shape  : Account = circle | Device = square | IP network = triangle | Delivery address = hexagon |
                Personal ID = diamond
  account fill: risk tier   LOW green | MEDIUM yellow | HIGH orange | CRITICAL red
  edge width  : number of transactions between the account and that entity
  dashed edge : account is verified under that personal ID
  layout      : every connected component (= possible fraud ring / one household) is laid out on its own
                and the components are packed side by side, biggest first.
"""
import math
import numpy as np
import networkx as nx
import pandas as pd

from .config import RELATIONS, TIER_COLORS, TIER_NAMES

REL_PREFIX = {"device": "D", "ip": "I", "address": "R"}
REL_STYLE = {"device": ("s", "#1565C0", "Device"), "ip": ("^", "#7E57C2", "Network"),
             "address": ("h", "#00897B", "Address")}
PERSON_STYLE = ("D", "#222222", "Personal ID")


def ego_accounts(res, i, hops=2, cap=25, max_accounts=150):
    """Accounts near account i: same-identity siblings first, then accounts that share a device / network / address.
    Through a hub entity (thousands of accounts) only the `cap` highest-risk members are taken, and the total is bounded
    by `max_accounts`, so the picture of one account stays readable and fast whatever the size of the data set."""
    gi = res.gi
    seen, cur = [i], [i]
    seen_set = {i}

    def add(m):
        if m not in seen_set and len(seen) < max_accounts:
            seen_set.add(m); seen.append(m); return True
        return False
    for _ in range(hops):
        nxt = []
        for a in cur:
            for m in gi.sib_nbr[gi.sib_ptr[a]:gi.sib_ptr[a + 1]]:
                if add(int(m)):
                    nxt.append(int(m))
        for a in cur:
            for R in gi.rel:
                for e in R["a_ent"][R["a_ptr"][a]:R["a_ptr"][a + 1]]:
                    mem = R["e_acct"][R["e_ptr"][e]:R["e_ptr"][e + 1]]
                    if len(mem) > cap:
                        mem = mem[np.argsort(-res.prob[mem])[:cap]]
                    for m in mem:
                        if add(int(m)):
                            nxt.append(int(m))
        cur = nxt
    return np.array(sorted(seen_set))


def build_nx(res, table, rels=RELATIONS, include_person=True, subset=None):
    gd = res.gd
    sub = np.arange(gd.n) if subset is None else np.asarray(subset)
    tier = dict(zip(table.account_id, table.risk_tier))
    score = dict(zip(table.account_id, table.risk_score))
    pred = dict(zip(table.account_id, table.prediction))
    G = nx.Graph()
    for i in sub:
        a = gd.account_ids[i]
        G.add_node("A:" + a, kind="account", name=a, tier=tier[a], risk=score[a], pred=pred[a],
                   person=gd.person_ids[i], n_events=int(round(np.expm1(gd.stats[i, 0]))), label=int(gd.labels[i]))
        if include_person:
            pn = "P:" + gd.person_ids[i]
            if pn not in G:
                G.add_node(pn, kind="person", name=gd.person_ids[i])
            G.add_edge("A:" + a, pn, kind="person", w=1)
    keep = np.zeros(gd.n, bool); keep[sub] = True
    for rel in rels:
        ed = gd.edges[rel]
        names = gd.ent_names[rel]
        m = keep[ed["src"]]
        for s_, d_, f in zip(ed["src"][m], ed["dst"][m], ed["feat"][m]):
            en = f"{REL_PREFIX[rel]}:{names[d_]}"
            if en not in G:
                G.add_node(en, kind=rel, name=str(names[d_]))
            G.add_edge("A:" + gd.account_ids[s_], en, kind=rel, w=int(round(np.expm1(f[0]))))
    return G


def select_subgraph(G, mode="all", focus=None, hops=2, max_nodes=900, collapse_private=False, min_tier="HIGH"):
    """mode: all | risky | focus.  Returns (subgraph, note)."""
    note = ""
    H = G
    if collapse_private:
        drop = [n for n, d in G.nodes(data=True) if d["kind"] not in ("account", "person") and G.degree(n) <= 1]
        H = G.subgraph([n for n in G if n not in set(drop)]).copy()
    if mode == "focus" and focus:
        f = "A:" + focus
        if f not in H:
            f = next(iter(H.nodes))
        H = nx.ego_graph(H, f, radius=hops * 2)
        return H, f"ego network of {focus} ({hops} hops)"
    comps = list(nx.connected_components(H))
    thr = TIER_NAMES.index(min_tier)

    def crisk(c):
        r = [H.nodes[n]["risk"] for n in c if H.nodes[n]["kind"] == "account"]
        return max(r) if r else 0

    def nrisky(c):
        return sum(1 for n in c if H.nodes[n]["kind"] == "account" and TIER_NAMES.index(H.nodes[n]["tier"]) >= thr)

    if mode == "risky":
        comps = [c for c in comps if nrisky(c) > 0]
        multi = [c for c in comps if len(c) > 2]          # an account + its identity only is not a 'group'
        comps = multi or comps
        note = f"connected groups containing >=1 {min_tier}/CRITICAL account"
    comps.sort(key=lambda c: (-nrisky(c), -crisk(c), -len(c)))
    keep, total = [], 0
    for c in comps:
        if total + len(c) > max_nodes and keep:
            continue
        if len(c) > max_nodes:
            continue
        keep.append(c); total += len(c)
    if not keep and comps:
        sub = H.subgraph(max(comps, key=nrisky)).copy()
        # biggest group is larger than the budget -> its highest-risk core
        core = sorted([n for n in sub if sub.nodes[n]["kind"] == "account"], key=lambda n: -sub.nodes[n]["risk"])[:max_nodes // 4]
        nodes = set(core)
        for n in core:
            nodes |= set(sub.neighbors(n))
        return sub.subgraph(nodes).copy(), "very large group - showing its highest-risk core"
    shown = sum(len(c) for c in keep); allc = sum(len(c) for c in comps)
    if shown < allc:
        note = (note + "; " if note else "") + f"showing {shown} of {allc} nodes (highest-risk groups first)"
    nodes = set().union(*keep) if keep else set()
    return H.subgraph(nodes).copy(), note or "full graph"


def layout_components(H, seed=1, hub_cap=10):
    """Components are laid out separately and packed. Shared 'hub' entities (> hub_cap accounts: public WiFi,
    office networks, shared phones) would glue unrelated households into one hairball, so they are removed
    for the layout and then placed at the centroid of the accounts they connect."""
    hubs = [n for n, d in H.nodes(data=True) if d["kind"] in RELATIONS and H.degree(n) > hub_cap]
    if hubs:
        L = H.copy(); L.remove_nodes_from(hubs)
        pos = _layout_components(L, seed)
        rng = np.random.default_rng(seed)
        for h in hubs:
            nb = [pos[m] for m in H.neighbors(h) if m in pos]
            pos[h] = (np.mean(nb, 0) if nb else np.zeros(2)) + rng.normal(0, 0.05, 2)
        return pos
    return _layout_components(H, seed)


def _layout_components(H, seed=1):
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    sizes = [max(1.8, 1.7 * math.sqrt(len(c))) for c in comps]
    W = max(math.sqrt(sum(s * s for s in sizes)) * 1.35, max(sizes) * 1.05, 3.0)
    pos, x, y, rowh = {}, 0.0, 0.0, 0.0
    for c, s in zip(comps, sizes):
        if x + s > W and x > 0:
            x, y, rowh = 0.0, y - rowh * 1.08, 0.0
        sub = H.subgraph(c)
        if len(c) == 1:
            p = {next(iter(c)): np.zeros(2)}
        elif len(c) == 2:
            a, b = list(c); p = {a: np.array([-.5, 0]), b: np.array([.5, 0])}
        else:
            p = nx.spring_layout(sub, k=1.3 / math.sqrt(len(c)), iterations=120 if len(c) < 600 else 50, seed=seed)
        arr = np.array(list(p.values()))
        ctr = arr.mean(0); span = max(np.abs(arr - ctr).max() * 2, 1e-6)
        for n, v in p.items():
            pos[n] = np.array([x + s / 2, y - s / 2]) + (v - ctr) / span * s * 0.92
        x += s * 1.08; rowh = max(rowh, s)
    return pos


def _hover(d, H, n):
    if d["kind"] == "account":
        return (f"<b>{d['name']}</b><br>risk {d['risk']:.0f}/100 ({d['tier']})<br>prediction: {d['pred']}"
                f"<br>personal ID: {d['person']}<br>events: {d['n_events']}<br>linked entities: {H.degree(n)}")
    kind = {"device": "Device", "ip": "IP network", "address": "Address", "person": "Personal ID"}[d["kind"]]
    nb = [m for m in H.neighbors(n) if H.nodes[m]["kind"] == "account"]
    return f"<b>{kind}</b> {d['name']}<br>linked accounts: {len(nb)}"


def draw_matplotlib(H, pos, title="", note="", focus=None, figsize=None, label_entities=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    n = len(H)
    side = float(np.clip(7 + n / 70, 9, 22))
    fig, ax = plt.subplots(figsize=figsize or (side * 1.15, side), dpi=110, facecolor="white")
    ax.set_facecolor("#F5F9FE")
    # edges
    from matplotlib.collections import LineCollection
    hubset = {n for n, d in H.nodes(data=True) if d["kind"] in RELATIONS and H.degree(n) > 10}
    for is_person in (True, False, "hub"):
        if is_person == "hub":
            es = [(u, v, d) for u, v, d in H.edges(data=True) if d["kind"] != "person" and (u in hubset or v in hubset)]
        elif is_person:
            es = [(u, v, d) for u, v, d in H.edges(data=True) if d["kind"] == "person"]
        else:
            es = [(u, v, d) for u, v, d in H.edges(data=True) if d["kind"] != "person" and u not in hubset and v not in hubset]
        if es:
            segs = [[pos[u], pos[v]] for u, v, d in es]
            lws = [0.6 if is_person is True else 0.5 + 0.55 * math.log1p(d["w"]) for u, v, d in es]
            ax.add_collection(LineCollection(segs, linewidths=lws, colors="#8FB0DC" if is_person is True else "#3F6AA6",
                                             linestyles=(0, (3, 3)) if is_person is True else "solid",
                                             alpha={True: 0.35, False: 0.45, "hub": 0.10}[is_person], zorder=1))
    pts = np.array(list(pos.values())) if pos else np.zeros((1, 2))
    ax.set_xlim(pts[:, 0].min() - 0.5, pts[:, 0].max() + 0.5); ax.set_ylim(pts[:, 1].min() - 0.5, pts[:, 1].max() + 0.5)
    deg = dict(H.degree())
    for rel in RELATIONS:
        m, col, _ = REL_STYLE[rel]
        ns = [x for x, d in H.nodes(data=True) if d["kind"] == rel]
        if ns:
            ax.scatter([pos[x][0] for x in ns], [pos[x][1] for x in ns], s=[40 + 9 * deg[x] for x in ns],
                       marker=m, c=col, edgecolors="white", linewidths=0.6, zorder=2, alpha=0.95)
    ps = [x for x, d in H.nodes(data=True) if d["kind"] == "person"]
    if ps:
        ax.scatter([pos[x][0] for x in ps], [pos[x][1] for x in ps], s=45, marker="D", c="#222", zorder=2)
    accs = [x for x, d in H.nodes(data=True) if d["kind"] == "account"]
    if accs:
        ax.scatter([pos[x][0] for x in accs], [pos[x][1] for x in accs], s=[70 + 14 * deg[x] for x in accs], marker="o",
                   c=[TIER_COLORS[H.nodes[x]["tier"]] for x in accs], edgecolors="black", linewidths=0.8, zorder=3)
    # labels
    for x in accs:
        d = H.nodes[x]
        if d["tier"] in ("HIGH", "CRITICAL") or (focus and x == "A:" + focus):
            if n <= 400 or (focus and x == "A:" + focus) or d["tier"] == "CRITICAL":
                ax.annotate(d["name"].replace("ACC_", ""), pos[x], xytext=(0, 9), textcoords="offset points",
                            fontsize=7, ha="center", weight="bold", zorder=5,
                            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    if label_entities and n <= 400:
        for x, d in H.nodes(data=True):
            if d["kind"] in RELATIONS and deg[x] >= 3:
                ax.annotate(d["name"][-8:], pos[x], xytext=(0, -9), textcoords="offset points", fontsize=5.5,
                            ha="center", color="#333", zorder=4)
    if focus and "A:" + focus in pos:
        p = pos["A:" + focus]
        ax.scatter([p[0]], [p[1]], s=900, facecolors="none", edgecolors="#1f77b4", linewidths=2.4, zorder=6)
    handles = [Line2D([], [], marker="o", ls="", mfc=TIER_COLORS[t], mec="black", ms=9, label=f"Account - {t} risk") for t in TIER_NAMES]
    for rel in RELATIONS:
        m, col, lab = REL_STYLE[rel]
        handles.append(Line2D([], [], marker=m, ls="", mfc=col, mec="white", ms=9, label=lab))
    handles.append(Line2D([], [], marker="D", ls="", mfc="#222", mec="#222", ms=7, label="Personal ID"))
    handles.append(Line2D([], [], color="#555", lw=2, label="transactions (width = count)"))
    handles.append(Line2D([], [], color="#999", lw=1, ls="--", label="account verified under ID"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1), fontsize=8.5, frameon=True, title="Legend")
    n_acc = len(accs); n_hi = sum(1 for x in accs if H.nodes[x]["tier"] in ("HIGH", "CRITICAL"))
    ax.set_title(f"{title}\n{n_acc} accounts ({n_hi} HIGH/CRITICAL)  |  {sum(1 for _,d in H.nodes(data=True) if d['kind'] in RELATIONS)} "
                 f"devices/IPs/addresses  |  {H.number_of_edges()} links" + (f"\n{note}" if note else ""), fontsize=11, color="#0D47A1")
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    return fig


def draw_plotly(H, pos, title="", note=""):
    import plotly.graph_objects as go
    ex, ey = [], []
    for u, v in H.edges():
        ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.6, color="rgba(110,110,110,0.45)"),
                               hoverinfo="skip", showlegend=False))
    symbols = {"device": "square", "ip": "triangle-up", "address": "hexagon", "person": "diamond"}
    colors = {"device": "#1565C0", "ip": "#7E57C2", "address": "#00897B", "person": "#263238"}
    names = {"device": "Device", "ip": "IP network (/24)", "address": "Delivery address", "person": "Personal ID"}
    for kind in ["person"] + RELATIONS:
        ns = [x for x, d in H.nodes(data=True) if d["kind"] == kind]
        if ns:
            fig.add_trace(go.Scatter(x=[pos[x][0] for x in ns], y=[pos[x][1] for x in ns], mode="markers", name=names[kind],
                                     marker=dict(symbol=symbols[kind], size=9 if kind != "person" else 8, color=colors[kind],
                                                 line=dict(width=0.5, color="white")),
                                     hovertext=[_hover(H.nodes[x], H, x) for x in ns], hoverinfo="text"))
    for t in TIER_NAMES:
        ns = [x for x, d in H.nodes(data=True) if d["kind"] == "account" and d["tier"] == t]
        if ns:
            fig.add_trace(go.Scatter(x=[pos[x][0] for x in ns], y=[pos[x][1] for x in ns], mode="markers",
                                     name=f"Account - {t} risk",
                                     marker=dict(symbol="circle", size=[11 + min(H.degree(x), 10) for x in ns],
                                                 color=TIER_COLORS[t], line=dict(width=1, color="black")),
                                     hovertext=[_hover(H.nodes[x], H, x) for x in ns], hoverinfo="text"))
    fig.update_layout(title=dict(text=f"{title}<br><sup>{note}</sup>"), height=760, showlegend=True,
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      margin=dict(l=5, r=5, t=70, b=5), plot_bgcolor="#F5F9FE", legend=dict(orientation="v"))
    return fig


def render_graph(res, table, mode="all", focus=None, hops=2, max_nodes=900, rels=RELATIONS, include_person=True,
                 collapse_private=None, min_tier="HIGH"):
    """One call -> (networkx subgraph, positions, note). Draw it with draw_matplotlib / draw_plotly.
    Large inputs are never drawn whole: 'focus' uses the account's neighbourhood, 'all' / 'risky' use the highest-risk
    accounts that fit into the node budget (the note says exactly how many of how many are shown)."""
    gd = res.gd
    subset, pre = None, ""
    if mode == "focus" and focus:
        subset = ego_accounts(res, gd.acct_index[focus], hops)
    elif gd.n > max_nodes // 3:
        k = max(10, max_nodes // 3)
        order = np.argsort(-res.prob)
        if mode == "risky":
            tiers = table.set_index("account_id").loc[gd.account_ids[order], "risk_tier"].to_numpy()
            order = order[np.isin(tiers, TIER_NAMES[TIER_NAMES.index(min_tier):])]
        subset = order[:k]
        pre = f"{len(subset):,} of {gd.n:,} accounts (highest risk first)"
    G = build_nx(res, table, rels, include_person, subset)
    if collapse_private is None:
        collapse_private = mode != "focus" and len(G) > max_nodes
    H, note = select_subgraph(G, mode, focus, hops, max_nodes, collapse_private, min_tier)
    if collapse_private:
        note += " | single-account entities hidden for clarity"
    if pre:
        note = pre + " | " + note
    return H, layout_components(H), note
