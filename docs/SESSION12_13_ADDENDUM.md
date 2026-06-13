# SESSION 12-13 ADDENDUM — the calibrated landscape arc (June 12, 2026)

Working dir: /home/claude/work/realNS_session9. Continues SESSION11_ADDENDUM.md.
Operator trust anchors unchanged (ns_part12_gate.py, ns_part3k.py rev K).

## Headline results

1. **Calibrated residual-landscape survey of the right half-plane (odd
   axisymmetric sector), tapered decay-adapted div-free basis (N = 5366):**
   - Real axis [-0.10, +0.30], step 5e-3 then 1e-3 near lam1: ONE smooth
     basin, unique minimum; **vertex at lambda = +0.113142** vs known
     lambda1 = 0.11314203 — agreement to 6 significant figures.
   - Valley value r(lam1) = 9.3141e-3; **certified pointwise** at 9.3141e-3
     (explicit field + one direct Lcols + grid gradient deflation; jitter
     1e-10/1e-12 stable) — independent of the gram pipeline.
   - Instrument noise 1.5e-4 RELATIVE (raw r^2 stable to 5 digits across
     M-cuts 1e-10/1e-12); local dips >= ~1% of backdrop are detectable.
   - NO second local minimum on the real axis.
   - Complex grid Re in [0,0.35] x Im in [0.05,0.60], step 0.05:
     COMPLETE (nsx_x5_grid.npz). All 8 rows STRICTLY MONOTONE increasing
     in Im; row minima track the real-axis basin; grid min 1.235e-2 at
     (0.10, 0.05) adjacent to the valley. NO off-axis local minimum:
     no candidate complex-conjugate unstable pair in the window.

2. **lambda1 independent confirmation**: Ritz |d| down to 4.53e-5
   (x4 masked basis, cut 1e-10); landscape vertex to ~1e-6.

3. **No second unstable Ritz value in any of FOUR basis layouts**
   (x1 joint-whitened, x4 split, x4-masked, x5 tapered). The next-highest
   Re Ritz value drifts with layout (-0.063, -0.13, -0.115): subspace-
   dependent, NOT claimed as an eigenvalue.

## The two lemmas (numerically demonstrated, writeup-ready)

L1 (open-quarter-domain): for exactly div-free decaying test fields, the
   pressure term <w, grad p> over the truncated quarter domain equals the
   boundary flux through the cut edges EXACTLY (closure 2.3e-13); the
   theta = pi/2 - 0.04 equatorial standoff carries 100% of it (p is odd
   about the equator: p/eps -> -126.69). Open-endpoint GL quadrature on
   (0, pi/2)^2 kills all four fluxes analytically: ||d|| collapsed
   6.5e4 -> 1.3e-8 (4e-13 relative). This made the pressure-free pencil
   exact and produced the first X1 PASS.

L2 (operator domain): radial families with R(0) != 0 give u_rho ~ P_k(mu)
   at the origin; the pencil <w, Lw'> converges but ||Lw||^2 ~ int rho^-2
   DIVERGES — the operator-side Gram AA does not exist (observed: AA diag
   6e18; high-n shell-Hermite are O(0.1) at rho=0 since psi_n(-8) is inside
   the turning point). Cure: origin mask m(rho) = rho^2/(rho^2+4) on ALL
   radial functions (preserves the div-free Stokes-stream construction);
   AA finite, bottom(Ad~) positive and cut-stable.

## Resolving-power law (the methods core)

   landscape floor ~= (eigenfunction best-approximation error in the span)
                      x (operator amplification of the missing content)
Measured chain: fit 4.4e-3 -> floor 0.066 (flat, no valley);
fit 2.56e-4 -> floor 9.3e-3 (valley appears, vertex at lam1 to 1e-6).
Explains session-11 run22's 1.1e-5 floor (phi-basis contains v1 exactly).
Calibration protocol: the known lam1 valley fixes the depth and location
standards for any candidate lam2 dip.

## Spectral design loop (how fit 4.4e-3 -> 2.56e-4 at SMALLER N)

- Exact Parseval spectroscopy of v1 on the open quarter (parity makes the
  odd-degree Legendre families orthogonal; gates 1.00000000): angular tail
  crosses 3e-4 at k=102, 3.6e-5 by k=120. The old 4.4e-3 "plateau" was
  100% angular truncation at K=78.
- Per-k constrained radial fits (R drives u1, D=2R+rho R' drives u2,
  jointly): single-sigma Hermite radial floor 8.4e-4; log-shell lattice
  in ln(rho) breaks it in union (5.1e-4); DUAL-SIGMA Hermite
  (sig=2.5 (+) sig=1.2) wins outright: floor 7e-5.
- Taper (heterogeneous tensor blocks; per-block CHOLESKY orthonormalization
  so prefixes nest): full-96 radial x k<=38, slim-46 x 40<=k<=120,
  tor-30 x k<=103 => N = 5366; X0r fit = 2.557e-4 (predicted 2.5e-4).
- T6 gate (heterogeneous G/H/AA assembly vs direct Lcols): 5.8e-16 /
  4.8e-16 / 4.7e-16 PASS. X1 PASS both cuts (|d| = 4.2e-4 / 3.0e-4).

## Dead ends with root cause (forensics, keep for the paper)

- Gram-Schur P_perp pipeline on the un-regularized basis: bottom(Ad~) =
  -0.011 (float64 cancellation; Lw ~99.999% gradient). Fixed by L2 + taper.
- Noisy tame-direction selection (nsx_x3): selection inherits Schur noise
  ~ tamest curvatures; flat landscape. Superseded.
- LOBPCG returned false negatives at lam1 (cold start). All production
  numbers use dense eigh subsets.
- Single-scale Chebyshev-in-beta family imploded numerically (untraced;
  moot). Fine log-shell lattice alone: 3.45e-3 (shell under-resolution).

## File inventory (this arc)

- nsx_x5.py (basis+gates+grams; modes gate|full), nsx_x5_grams.npz (1.26GB:
  G,G0,H,AA,B,sizes), nsx_x5b.py/.log (Schur+real sweep),
  nsx_x5_prep.npz (Gt, At_10, At_12), nsx_x5_real.npz,
  nsx_x5c.py/.log + nsx_x5_grid_ck.npz (complex grid, IN FLIGHT),
  nsx_x5r.py/.log (pointwise certifier, DONE: 9.3141e-3),
  nsx_x0e/f.py/.log (fit scans + spectroscopy + family shootout),
  nsx_x4*.py/.log (masked split basis), nsx_diag2-5.py (L1 forensics),
  nsx_x2_aa.py (T5-gated aa_block/b_block kernels, reused by x5),
  /mnt/user-data/outputs/landscape_x5_real.pdf (new figure).

## COMPLETED THIS ARC

- Complex grid done; two-panel figure landscape_x5_full.pdf in outputs.
- **Preprint written and shipped (now labeled v1.1, 12 pp.)**
  (preprint_survey_v4.tex/.pdf): new Section 6 (rebuild + Lemma
  open-domain + operator-domain Remark + resolving-power law eq. +
  design loop + calibrated survey), gates table (tab:gates), updated
  abstract/intro/limitations, LS04/TE05 bib entries. Remaining TODOs in
  tex: author block, HWY permissions, repo link (campaign owner items).
- Bundle realNS_session13_full.tar.gz in outputs.

## Next session (if any)

Optional hardening: even-sector survey on a rebuilt basis (the one
sector gap closable with this machinery); finer grid near the real axis
(Im < 0.05); or rigorous-enclosure exploration. Otherwise the campaign
deliverable is complete.
