# 12 equations and reference implementation

The public `reproducibility/aig_osr_reference.py` preserves the metric computation path of the locally audited reference implementation. Renaming the method and removing the unrelated legacy 3D point-densification export do not alter Eqs. (1)–(12).

| Equation | Mathematical role | Code |
|---|---|---|
| (1) | Baseline plus denormalized residual | `dz_mean`, `dz_std`, and `z_net = base_t + ...` |
| (2)-(5) | Six-channel conditionally modulated three-layer periodic neural field | `ConditionedPeriodicField` and `PeriodicLayer` |
| (6) | MBES fit, mask-weighted baseline and gradient losses | `loss_mbes`, `loss_base0`, `loss_smooth` in the training loop |
| (7)-(8) | Distance-weighted local Gaussian reference | `adaptive_confidence_smoothing` |
| (9) | Eight-neighbour graph weights and weak cross-mask links | `build_edge_aware_laplacian_8n` |
| (10) | Squared-Laplacian correction with hard controls | `hard_dirichlet_projection` |
| (11) | Free-cell sparse linear system | `A_ff`, `A_fc`, `b_f` and conjugate-gradient solve |
| (12) | Raster-distance observation support | `hole_px` and `apply_topology_hole_digging` |

Every tenth step the program also computes `loss_illum_phys`, but it evaluates the predicted image inside `torch.no_grad()`. Adding that scalar to the displayed loss does not affect model parameters. It is deliberately excluded from the effective objective in Eq. (6).

**Scope of the audit:** the equations and this reference code agree in algorithmic structure and default parameters. This does not identify which historical source version produced each archived figure or table. The published sample is synthetic and cannot validate the paper's field accuracy claims.
