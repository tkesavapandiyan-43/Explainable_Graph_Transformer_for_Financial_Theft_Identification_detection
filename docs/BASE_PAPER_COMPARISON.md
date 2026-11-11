> Note (final version): numbers quoted in this document come from the first demo model; the final comparison is in `RESULTS_AND_PANEL_NOTES.txt` and the GUI.

## Base paper vs this project

**Base paper:** M. Gao et al., *"Fine-Grained Behavioral Modeling With Graph Neural Networks for Financial Identity Theft Detection"* (EnvIT), IEEE Trans. Network Science and Engineering, vol. 13, 2026.
**This project:** *Explainable Graph Transformer (EGT) for Financial Identity Theft Detection.*

> **Comparability:** the paper trains on a private Meituan data set (and on Vesta with its own account construction and random 3:1:1 split).
> The shipped demo model is trained on GENERATED data that imitates the paper's fraud patterns; the real-data command trains on Vesta / IEEE-CIS with a time-ordered split. **Numbers must not be compared 1-to-1 with the paper's numbers.**
> Compare models only inside `outputs/benchmark_table.csv`, where every model uses the same split.

### Module by module
| # | Module | Base paper (EnvIT) | This project (EGT) | Same function? | Status / reason |
|---|---|---|---|---|---|
| 1 | Environmental aggregation | GAT over account-account neighbours linked by shared device / IP / location (meta-paths A-D-A, A-I-A, A-L-A), 8 heads, summed over meta-paths | Heterogeneous **graph-transformer attention** account→entity→account for each relation (device, network, address) + HAN-style **relation-level attention**; 2 layers, 4 heads | Yes - same purpose, same data flow (a shared entity connects accounts) | **Both exist:** the paper's exact GAT is in `egt/envit.py` (benchmark row "EnvIT (base paper)"); our improved version is `model.py: EnvAggLayer, SparseAttn`. Entities are real nodes, so a hub is a bounded set of attention edges instead of a quadratic account-account clique; the neighbour set can be sampled for scalability. Used in the final prediction (`tests/` checks gradient flow). |
| 2 | Environmental difference | GraphSAGE (mean) on the per-relation account-entity sub-graphs, then GraphSAGE (mean) on the account graph linking accounts with the same personal ID | **Edge-aware graph-transformer attention** on the same-identity account graph; each edge carries 8 *difference features* (Jaccard overlap of devices / networks / addresses, gaps in their counts, night-share gap). Initialised with the aggregation embedding | Yes - compares sibling accounts' environments | **Both exist:** the paper's GraphSAGE-mean is in `egt/envit.py`; our improved version is `EnvDiffLayer`. Mean aggregation cannot tell *how different* a sibling is; here the difference modulates keys and values. Environment embedding is shared with module 1 instead of a second, separate bipartite pass. |
| 3 | Behaviour extractor | T-LSTM with g(Δt)=1/log(Δt+e), short/long-term memory decomposition; sum of hidden states | **Same T-LSTM equations**; Δt in minutes; hidden states pooled by **event-level attention** instead of a plain sum | Yes | **Exact core + improved pooling** (`TLSTM`, `BehaviourEncoder`). The event attention is part of the forward pass and is what the explanation module reports. |
| 4 | Attentive feature fusion | Scaled dot-product self-attention over 3 vectors (aggregation, difference, behaviour), rows summed | Multi-head self-attention (+ residual, LayerNorm, FFN) over **4 tokens** (profile, aggregation, difference, behaviour), then attention pooling → explicit module weights | Yes | **Improved** (`EGT.fuse`). Profile token carries the account statistics (the paper uses them as the initial embedding). Ablation "w/o attentive fusion" is in the benchmark. |
| 5 | Detection layer | Fully-connected + softmax; cross-entropy on labelled accounts | MLP + softmax; class-weighted cross-entropy on labelled accounts; probability then temperature-calibrated | Yes | **Same + calibration** |
| 6 | Semi-supervised use of unlabeled accounts | Unlabeled accounts take part in message passing only | Same design (loss only on labelled seeds; unlabeled nodes may be neighbours) | Yes | **Implemented.** (The Vesta files have no unlabeled accounts, so it is inactive on that data.) |
| 7 | Class imbalance | Under-sampling of normal accounts | Class-weighted loss by default; **under-sampling implemented** (`TrainConfig.undersample`) and benchmarked as "EGT + under-sampling" | Yes | **Implemented** - both strategies are run and reported. |
| 8 | Training protocol | Adam, lr 1e-3, batch 256, early stop 15 epochs, 5 runs, 3:1:1 random split | Adam, lr 1e-3, mini-batches of 512 seed accounts with neighbour sampling, early stopping (patience 10, configurable), 3 seeds, **time-ordered 60/20/20 split** (random 3:1:1 available via `prepare --split random`) | Mostly | **Adapted** for scale and leakage control; all preprocessing statistics are fitted on training accounts only. |
| 9 | Metrics | AUC, Recall@80/85/90 % precision, Welch's t-test | AUC, PR-AUC, Recall@80/85/90 %P, precision, recall, F1, accuracy, MCC, Brier, ECE; **Welch's t-test** over seeds | Yes | **Implemented** (`metrics.py`, `evaluate.py`). |
| 10 | Baselines | LSTM, Bi-LSTM, GRU, HAInt-LSTM, TSF, GraphSAGE, GAT, R-GCN, HAN, HACUD, SOBT, SEC-GFD | **EnvIT itself (the paper's model, re-implemented)**, logistic regression, random forest, gradient boosting (profile-only and all-feature) + 4 ablations + under-sampling variant | Partly | The paper's other neural baselines (HAN, R-GCN, SEC-GFD ...) are not re-implemented; every listed model is a real run on the same split. |
| 11 | Interpretability | Average attention weight per meta-path (Fig. 5) | Per-account explanation: **SHAP**, Integrated Gradients, attention evidence, verdict | Extends | **Novel contribution** |
| 12 | Data | Meituan (private) + Vesta | Vesta / IEEE-CIS (public) only | - | Meituan unavailable. |

### Contributions
**Base-paper contributions (re-implemented):** heterogeneous + homogeneous graph view of accounts and their access environment; environment aggregation / difference / behaviour extractors; attentive fusion; semi-supervised graph training; R@T%P evaluation.

**Our contributions (implemented and tested):**
1. Transformer-style edge-aware attention in place of GAT / GraphSAGE (modules 1-2) and event-level attention pooling in the T-LSTM.
2. **Explainability** - **SHAP** feature-group attribution, Integrated-Gradients feature attribution, module counterfactuals, attention over shared devices / networks / addresses and sibling accounts, influential events, plain-English verdict (`explain.py`).
3. **Dynamic risk scoring** - temperature-calibrated probability → 0-100 risk score, validation-learnt threshold policies (F1, precision-90, recall-90, manual, top-K alert budget), auditable risk tiers, snapshot timeline (`risk.py`, `pipeline.py`).
4. **Scalable training** - neighbour-sampled mini-batches, chunked pre-processing of the complete Vesta files, deterministic sub-graph scoring (`sampler.py`, `vesta.py`).
5. Full-schema Vesta feature pipeline (transaction, card, address, e-mail, C/D/M/V, identity, device, OS, browser, screen).

### Algorithm status
* **GAT, GraphSAGE (paper):** implemented in `egt/envit.py` and trained as a baseline; the proposed EGT replaces them by edge-aware Graph-Transformer attention.
* **SHAP:** implemented in `egt/shap_explain.py` (permutation Shapley estimator, exact "efficiency" check in the tests and in the GUI).
* **LIME, GNNExplainer:** not used. LIME is a local surrogate without a faithfulness guarantee for graph models; GNNExplainer would learn an edge mask that our attention already exposes. SHAP + Integrated Gradients cover the same need with exact additivity.
* **Real-time streaming:** not implemented; the timeline is snapshot-based re-scoring.
