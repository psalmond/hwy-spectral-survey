# SESSION 11 ADDENDUM (FINAL) — lambda_2 candidate REFUTED; paper reframed

## Part 1 (pre-repo, from bundled pencils) — still valid
* Exact symmetry eigenvalues (analytic):
    L_U (d_z U) = -1/2 (d_z U)              [ODD]
    L_U (U + xi.grad U) = -1 (U + xi.grad U) [EVEN]
  Both rho^-2 decay => genuine L2 eigenfunctions. CORRECTION: ns_part3m.py
  comment "d_z U ... eigenvalue 1/2" has the wrong sign (-1/2 correct).
  No symmetry produces Re>=0 => instability-order proviso vacuous for L_U.
* Deflation anatomy from Gram: |<v1,v2>| = 0.97; phi-span holds 7.5% of
  v1; phi-only pencil keeps a positive value +0.032461 with vector
  orthogonal to v1 (|cos|=0.008).
* Novelty re-check: gap survives; IJP part II not posted; no HWY v3.

## Part 2 (repo re-uploaded; md5 gate PASS, 24+19 phi files)
* run21 (ns_part3p.py): pointwise residual of the PHI-ONLY deflated
  candidate vector: r = 3.952e-1 at its Ritz value (= its optimal lambda).
  JUNK-LEVEL (individual-phi floor 0.35). The candidate's r=3.9e-2 was
  carried entirely by its v1 component. Ambiguous alone (a genuine
  near-parallel eigenfunction predicts ~0.31 for the orthogonal part),
  so =>
* run22 (ns_part3q.py): RESIDUAL LANDSCAPE
  r_min(lam) = min_c ||(A - lam M)c|| / ||Mc|| over the 25-dim odd span,
  one streamed pass for AA/AM/MM (cached: run22_grams.npz). VERDICT:
  - single smooth valley centered lam* = +0.113144, floor 1.10e-5;
  - NO second dip anywhere in [-1.2, +0.18]; r_min(0.036)=2.3e-2 on a
    smooth flank; r_min(0)=3.2e-2; monotone left flank through -1/2
    (span does not represent d_z U);
  - flank slope => kappa_eff ~ 3.6; within-pencil biorthogonality
    kappa(lambda_1) ~ 10.5, kappa(0.0363) ~ 18.9. Shadow explanation
    requires only kappa >= ~2.
  => The +0.036 "candidate" is a NON-NORMAL PSEUDOSPECTRAL SHADOW of
  lambda_1. All its credentials (r=3.9e-2 single-vector residual,
  25/25 jackknife, realness, isolation) are quantitatively explained.
  *** lambda_2 CLAIM REFUTED at current evidence. Do NOT post v2. ***
* run23 (ns_part3r.py): EVEN landscape (run23_grams_even.npz):
  r_min >= 0.66 on all of [-1.5, +0.8] — the tail-heavy Phi span has NO
  spectral resolving power in truncated true L2. +0.51 = noise
  (strongest form of last session's artifact classification); no dip at
  the exact -1 either. Even sector: no spectral conclusion possible
  from this span, in either direction.
* s=0.54 inverse iteration: MOOT (seed was a shadow). Not run.

## Methodological capital (the new paper's spine)
The residual landscape r_min(lambda) is a cheap (one extra streamed
pass), decisive validity test for subspace spectral surveys of
non-normal operators. This session it unmasked a false positive that
had passed: calibration gate, negative control (U:=0), quadrature
stability, pointwise single-vector residual, AND 25/25 jackknife.
Recommend reporting it by default; relevant to shift-invert Krylov
practice (IJP 2606.07501, Wang et al 2509.14185).

## DELIVERABLE: preprint_survey_v3.tex/.pdf (+ landscape.pdf figure)
Reframed: "calibrated spectral survey ... symmetry eigenvalues, and the
anatomy of a pseudospectral false positive". Headline claims:
  (1) exact symmetry eigenvalues lemma (-1/2 odd, -1 even);
  (2) Galerkin obstruction (conditioning/band tradeoff);
  (3) negative survey: no evidence of a second unstable eigenvalue at
      subspace resolution (carefully bounded: evidence of absence only
      within the spans); instability order remains OPEN;
  (4) the landscape instrument + the false-positive anatomy (tables
      retained as exhibits of how convincing a shadow can look).
v2 (instability order >= 2 framing) is WITHDRAWN — superseded by v3.
Remaining TODOs: author block; HWY permissions/etiquette; code repo link.

## NEXT SESSION (if pursued)
1. Decide: post v3 as methods/negative-survey paper, or bank it and
   pursue a genuine survey instrument first (decay-adapted radial basis
   with rebuilt div-free slaving — full session + new gates; this is now
   the ONLY route to actually answering the instability-order question).
2. Optional rigor rung: validate Lemma via pointwise residual of
   FD-built d_z U at lam=-1/2 (needs FD pressure d_z P from UP.mat).
3. Read HWY Julia src for certified finite-rank design (unchanged).
4. gCLM Summit B still banked separately, untouched.

## New files
ns_part3p.py / ns_part3_run21.log   deflated-vector pointwise residual
ns_part3q.py / ns_part3_run22.log   odd landscape + kappa (grams cached)
ns_part3r.py / ns_part3_run23.log   even landscape (grams cached)
ns_part3_run23a.log                 extended odd sweep (from cache)
run22_grams.npz / run23_grams_even.npz   pointwise Gram caches (small)
landscape.pdf                       two-panel figure (in paper)
preprint_survey_v3.tex/.pdf         reframed draft (SUPERSEDES v2)
