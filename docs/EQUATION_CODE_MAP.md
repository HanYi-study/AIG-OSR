# Equations and reference implementation

The public `reproducibility/aig_osr_reference.py` preserves the metric computation path of the locally audited reference implementation. Renaming the method and removing the unrelated legacy 3D point-densification export do not alter Eqs. (1)–(8).

| Equation | Mathematical role | Code |
|---|---|---|
| (1) | Baseline plus denormalized residual | `dz_mean`, `dz_std`, and `z_net = base_t + ...` |
| (2) | Six-channel FiLM-conditioned three-layer SIREN | `CondFiLMSIREN` and `SineLayer` |
| (3) | MBES fit, mask-weighted baseline and gradient losses | `loss_mbes`, `loss_base0`, `loss_smooth` in the training loop |
| (4) | Distance-weighted local Gaussian reference | `adaptive_confidence_smoothing` |
| (5) | Eight-neighbour graph weights and weak cross-mask links | `build_edge_aware_laplacian_8n` |
| (6) | Squared-Laplacian correction with hard controls | `hard_dirichlet_projection` |
| (7) | Free-cell sparse linear system | `A_ff`, `A_fc`, `b_f` and conjugate-gradient solve |
| (8) | Raster-distance observation support | `hole_px` and `apply_topology_hole_digging` |

Every tenth step the program also computes `loss_illum_phys`, but it evaluates the predicted image inside `torch.no_grad()`. Adding that scalar to the displayed loss does not affect model parameters. It is deliberately excluded from the effective objective in Eq. (3).

**Scope of the audit:** the equations and this reference code agree in algorithmic structure and default parameters. This does not identify which historical source version produced each archived figure or table. The published sample is synthetic and cannot validate the paper's field accuracy claims.
