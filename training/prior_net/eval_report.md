# Prior Net v0 — held-out eval

- model: `training/prior_net/prior_net_v0.pt` (3,461,961 params)
- train/val rows: 95000 / 5000
- best epoch: 14, val loss 0.5893
- full-description exact match: **1.9%**

| head group | accuracy |
|---|---|
| body_init | 75.0% |
| cmp | 89.2% |
| const | 41.9% |
| else | 88.7% |
| gate | 85.7% |
| op | 91.2% |
| ret | 67.9% |
| src | 85.6% |

