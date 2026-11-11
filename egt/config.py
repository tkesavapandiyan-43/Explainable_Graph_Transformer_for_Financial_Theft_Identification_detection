"""Central configuration, constants and column schema."""
from dataclasses import dataclass, asdict

# Input schema: one row = one transaction / activity event
REQUIRED_COLS = ["account_id", "timestamp"]
OPTIONAL_COLS = ["person_id", "event_type", "device_id", "ip", "address", "amount", "label"]

RELATIONS = ["device", "ip", "address"]          # environmental entity types (paper: M = 3)
REL_COLUMN = {"device": "device_id", "ip": "ip", "address": "address"}

SEQ_LEN = 40                                     # number of time steps (paper: 40)
SEQ_DIM = 12                                     # per-event feature size

TIER_NAMES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
TIER_COLORS = {"LOW": "#2e9e5b", "MEDIUM": "#f2c230", "HIGH": "#f08a24", "CRITICAL": "#d62728"}


@dataclass
class ModelConfig:
    arch: str = "egt"              # 'egt' = proposed Explainable Graph Transformer | 'envit' = base paper re-implemented (GAT + GraphSAGE)
    n_stats: int = 0               # filled automatically
    seq_dim: int = SEQ_DIM
    hidden: int = 48
    heads: int = 4
    env_layers: int = 2
    diff_layers: int = 2
    dropout: float = 0.25
    edge_dim: int = 2
    sib_edge_dim: int = 8
    n_rel: int = len(RELATIONS)
    # ablation switches
    use_agg: bool = True
    use_diff: bool = True
    use_beh: bool = True
    use_fusion_attn: bool = True
    rel_mask: tuple = (1, 1, 1)    # meta-path study: which of device / ip / address relations are used

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    lr: float = 1e-3               # paper
    weight_decay: float = 1e-5
    epochs: int = 120
    patience: int = 15             # paper: early stopping after 15 epochs w/o val-AUC gain
    seed: int = 42
    split: tuple = (3, 1, 1)       # paper: train : val : test = 3 : 1 : 1 (used when no pre-computed split exists)
    batch_size: int = 512          # seed accounts per mini-batch
    cap_acct_per_ent: int = 20     # neighbour sampling: other accounts kept per device / network / address
    cap_sib: int = 10              # neighbour sampling: sibling accounts (same personal ID) kept per account
    undersample: int = 0           # paper's under-sampling: keep this many normal accounts per fraud account per epoch (0 = off)
    max_batches: int = 0           # optional cap on mini-batches per epoch (0 = whole training set every epoch)
    max_minutes: float = 0.0       # wall-clock budget per training run (0 = unlimited); the best epoch so far is kept
    weight_clip: float = 10.0      # cap of the fraud-class loss weight (used when under-sampling is off)
