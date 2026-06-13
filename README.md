# hwy-spectral-survey
Pre-print DOI: https://doi.org/10.5281/zenodo.20675225


A calibrated spectral survey of the linearized Navier–Stokes operator at
the Hou–Wang–Yang (HWY) self-similar profile
([arXiv:2509.25116](https://arxiv.org/abs/2509.25116)): symmetry
eigenvalues, the anatomy of a pseudospectral false positive, and a
calibrated residual-landscape survey on a from-scratch decay-adapted
divergence-free basis.

**Paper:** `paper/preprint_survey_v4.tex` (P. Salmond, draft PDF, https://doi.org/10.5281/zenodo.20675225).

## Headline results

| Quantity | Value |
|---|---|
| Landscape vertex (real axis) | **λ = 0.113142** (known λ₁ = 0.1131420…, six-figure agreement) |
| Valley depth r(λ₁) | **9.3141 × 10⁻³**, certified pointwise by an independent path |
| Instrument noise | 1.5 × 10⁻⁴ *relative* (cut-stable) |
| Second minimum, real axis [−0.10, 0.30] | none |
| Second minimum, complex window [0, 0.35] × [0.05, 0.60]i | none (strictly monotone in Im λ on every row) |
| Exact symmetry eigenvalues | −1/2 (translation), −1 (time-translation) — both stable |
| v₁ fit error of the tapered trial space (N = 5366) | 2.557 × 10⁻⁴ |

Conclusion (evidence-grade, floating point, **odd axisymmetric sector
only**): no second unstable eigenvalue at the instrument's stated
sensitivity; the instability order of the HWY profile in this sector is
one. See the paper's Limitations section for scope.

## What's in here

```
src/        all Python (flat module layout; scripts import by name)
  ns_part12_gate.py     gate-validated pointwise operator realization
  ns_part3k.py, ns_part3f.py, ns_part3_spectrum.py
                        operator application (Lcols), geometry, quadrature
  hdf5min.py            self-contained pure-Python MATLAB v7.3 reader
  nsx_basis.py, nsx_op.py, nsx_x2_aa.py
                        Stokes-stream div-free basis; separable assembly
                        kernels (gates T4/T5)
  nsx_x5.py             production basis + grams + gates (modes: gate|full)
  nsx_x5b.py            Schur deflation + real-axis landscape
  nsx_x5c.py            complex-grid survey (checkpointed)
  nsx_x5r.py            pointwise certification of the landscape valley
  mk_heatmap.py         figures
  nsx_diag*.py, nsx_x0*.py, nsx_r1*.py, nsx_x4*.py, nsx_fitres.py
                        forensic diagnostics: the open-domain lemma
                        demonstrations, operator-domain divergence,
                        target spectroscopy, radial-family shootout,
                        resolving-power measurements (paper §6)
data/       NOT included — fetched from the HWY public repository
            (see data/README.md; no-redistribution note)
results/    small result arrays (landscapes) + figures; large Gram
            caches are regenerable (see docs/REPRODUCE.md)
paper/      LaTeX source + PDF
docs/       REPRODUCE.md (the gate ladder) + campaign lab notes
```

## Quick start

```bash
pip install -r requirements.txt
# fetch HWY data (see data/README.md), then:
cd src
python ns_part12_gate.py        # operator trust anchors
python nsx_x5.py gate           # T6 assembly gate (machine precision)
python nsx_x5.py full           # grams + X0r/X1 gates  (~7 min, ~2 GB)
python nsx_x5b.py               # Schur + real-axis landscape (~35 min)
python nsx_x5c.py               # complex grid (~80 min, checkpointed)
python nsx_x5r.py               # pointwise certification
python mk_heatmap.py            # figures
```

Full expected outputs, runtimes, and tolerances: `docs/REPRODUCE.md`.

## Attribution and original contributions

This repository and the accompanying preprint are the work of
**P. Salmond**. The following are introduced here and should be
attributed accordingly if reused or built upon (full statements and
derivations in `paper/preprint_survey_v4.pdf`):

- the **open-domain pressure lemma** — for divergence-free decaying
  fields, the weak-form pressure term equals the cut-edge flux exactly,
  vanishing under open-endpoint quadrature of the parity-reduced
  quarter domain but $O(1)$ under the standard equatorial standoff;
- the **operator-domain regularization** — radial factors with
  $R(0)\ne 0$ lie outside the operator domain
  ($\|\mathbf{L}_U w\|\notin L^2$); the mask $\rho^2/(\rho^2+a^2)$
  cures it while preserving divergence-freeness;
- the **resolving-power law and calibration protocol** for residual
  landscapes (floor $\approx$ eigenfunction best-approximation error
  $\times$ operator amplification), calibrated against the known
  eigenvalue;
- the **decay-adapted divergence-free trial space** (Stokes-stream
  family, dual-width radial functions, spectrum-designed taper) and the
  gate-validated assembly/landscape pipeline.

Reuse of code is governed by the MIT `LICENSE` (which requires keeping
the copyright notice); reuse of results or methods should additionally
**cite the preprint** (`CITATION.cff`). Please also cite Hou–Wang–Yang,
arXiv:2509.25116, whose profile and verified eigenpair this work builds
on.

## Provenance (what is theirs, what is ours)

**Theirs (Hou-Wang-Yang):** the data files of their public repository
(profile, certified eigenpair, phi-families) — fetched at run time,
**never redistributed here** — and all the mathematics of
arXiv:2509.25116. **Ours (MIT):** every line of code in `src/`
(an independent Python implementation of their published operator,
plus the bases, instruments, and diagnostics), all derived results in
`results/`, and the paper. The operator-layer modules mirror their
data formats and conventions (facts/interfaces), but contain no code
from their repository. Per-file SPDX headers and `NOTICE` make this
explicit.

## Data and licensing

The HWY repository carries no license file; **none of its data is
redistributed here**. The pipeline reads it from the authors' public
repository as published scientific record, with attribution. All code
in this repository is MIT-licensed; results in `results/` are derived
quantities.

## Citation

See `CITATION.cff`. Please also cite Hou–Wang–Yang, arXiv:2509.25116,
whose profile and verified eigenpair this survey builds on.
