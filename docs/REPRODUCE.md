# REPRODUCE.md — the gate ladder

Environment: Python 3.10+, numpy/scipy/matplotlib, 4 cores, 4 GB RAM
suffice for everything below. All runs from `src/`, with the HWY data
fetched per `data/README.md`. Every stage prints its own pass/fail
against the tolerances listed; do not proceed past a failed gate.

## 0. Operator trust anchors
`python ns_part12_gate.py`
- profile equation residual ≤ 1e-8 (campaign value 6.2e-9)
- stored eigenpair (λ₁ = 0.11314203…) residual ≤ 1e-4 (campaign 1.7e-5)

## 1. Assembly kernel gates (machine precision)
`python nsx_x5.py gate` — heterogeneous (tapered) block assembly of
G, H, AA vs. direct columnwise application of the validated operator:
expect ≤ 1e-11 (campaign: 5.8e-16 / 4.8e-16 / 4.7e-16).
The underlying kernels also carry their own gates (T4 in nsx_op usage,
T5 in `nsx_x2_aa.py gate`), each ≤ 1e-11.

## 2. Production grams + pencil gates  (~7 min, ~2 GB)
`python nsx_x5.py full` → `nsx_x5_grams.npz` (~1.3 GB). Expect:
- X0r: v1 fit error 2.56e-4 (±1 in the third digit)
- X1: λ₁ Ritz |d| ≤ 5e-3 at both cuts (campaign 4.2e-4 / 3.0e-4)
- X3: maxRe(U=0) ≤ −1/4 + 5e-3
- AA symmetry ≤ 1e-14 (campaign 2.1e-16)

## 3. Schur deflation + real-axis landscape  (~35 min, ≤1.8 GB)
`python nsx_x5b.py` → `nsx_x5_prep.npz`, `nsx_x5_real.npz`. Expect:
- bottom(Ãd) positive, stable to ≥5 digits across cuts 1e-10/1e-12
  (campaign +2.392e-4 / +2.393e-4)
- r(λ₁) = 9.31e-3, r(0) = 1.55e-2, r(0.25) = 3.28e-2
- single basin on [−0.10, 0.30]; fine vertex at λ = 0.113142

## 4. Complex grid  (~90 min, checkpointed)
`python nsx_x5c.py` → `nsx_x5_grid.npz` (resumes from
`nsx_x5_grid_ck.npz` if interrupted). Expect strict monotone increase
in Im λ on every row; grid min 1.24e-2 at (0.10, 0.05); no off-axis
local minimum.

## 5. Pointwise certification  (~5 min)
`python nsx_x5r.py` — rebuilds the landscape arg-min as an explicit
field, applies the operator once directly, deflates on the grid.
Expect 9.314e-3 at both jitters, matching step 3.

## 6. Figures
`python mk_heatmap.py` → `results/figures/landscape_x5_full.pdf`.

## Forensics (paper §6; optional)
- Open-domain lemma demonstration: `nsx_diag5.py` (edge-flux closure
  2.3e-13; equatorial standoff carries ratio 1.000), and the pre-flight
  in the lab notes (pressure term 6.5e4 → 1.3e-8 on the open domain).
- Operator-domain divergence: run `nsx_x4.py` with the origin mask
  removed (AA diag → ~1e18) vs. as shipped.
- Target spectroscopy + radial-family shootout: `nsx_x0f.py`
  (Parseval gates 1.00000000; angular tail 3e-4 at k=102; dual-σ
  radial floor 7e-5; log-shell union 5.1e-4).
- Resolving-power chain: `nsx_fitres.py`, `nsx_r1.py`, `nsx_r1b.py`
  (deflation-span calibration: control-B floor 7.6e-5).

Known environment notes: GL quadrature must use open endpoints
(0, π/2) — see paper Lemma 1; LOBPCG is not used in any production
number (documented cold-start false negatives); all eigensolves are
dense LAPACK.
