# Data (not redistributed)

This pipeline reads numerical data (the precomputed self-similar profile
and eigenpair) from the Hou–Wang–Yang work accompanying arXiv:2509.25116.
That material publishes no license file, so **none of it is
redistributed here**; obtain it yourself.

## Where to get it

The authors' canonical repository is:

  https://github.com/HouGroup2026/3d-navier-stokes-nonuniqueness

```
git clone https://github.com/HouGroup2026/3d-navier-stokes-nonuniqueness
```

**If that URL is unavailable** (repositories for in-review papers are
sometimes toggled private, renamed, or moved): the authoritative,
persistent pointer is the **arXiv abstract page itself** —
<https://arxiv.org/abs/2509.25116> — under its "Code & Data" / ancillary
links, which the authors keep current. Use whatever that page links to.
A co-author (Yixuan Wang) also keeps a personal fork
(`github.com/RoyWangyx/3d-navier-stokes-nonuniqueness`) — a trustworthy
copy if the org repo is briefly unavailable, but use the `HouGroup2026`
org as the source of record, since a personal account may be
reorganized at the author's discretion.

## A note on repository availability

As of **13 June 2026**, the canonical `HouGroup2026` organization repository
was intermittently unavailable, while the co-author's fork
(`github.com/RoyWangyx/3d-navier-stokes-nonuniqueness`) was reachable and
its `data/` folder contained the required `.mat` files. If the
`HouGroup2026` URL does not resolve when you try it, this is expected to
be transient — use the arXiv abstract page links or the co-author fork as
described above. Repository availability for in-review papers can change
without notice; the arXiv abstract page (arXiv:2509.25116) is the most
durable pointer.

## Pointing the code at the data

The profile/eigenpair data lives under that repository's `data/`
directory. Override the default path with the `HWY_REPO` environment
variable:

```
export HWY_REPO=/path/to/3d-navier-stokes-nonuniqueness
```

(The code's default is `../data/3d-navier-stokes-nonuniqueness`, i.e. a
clone placed next to this directory.)

## Data formats (two MATLAB variants)

The HWY `.mat` files come in two MATLAB on-disk formats, and the code
reads each with the appropriate loader — no MATLAB installation is
needed to run this pipeline:

- **`UP.mat`, `up_eig.mat`** (the profile and eigenpair) are classic
  MATLAB files (format v7 or earlier) and are read with
  `scipy.io.loadmat` (a listed dependency in `requirements.txt`).
- **`phi_odd/up_phi_*.mat`, `phi_even/UP_phi_*.mat`** (the auxiliary
  generalized-eigenfunction families, used only by `ns_part3k.py`) are
  MATLAB **v7.3** files, which are HDF5 under the hood. These are read
  by `src/hdf5min.py`, a small self-contained pure-Python reader for the
  v7.3 subset used here — so you do **not** need `h5py`, MATLAB, or any
  HDF5 system library.

If a future version of the upstream data saves `UP.mat`/`up_eig.mat` in
v7.3 instead of classic format, `scipy.io.loadmat` will raise a clear
error; in that case those files can be routed through `hdf5min.py` the
same way the `phi_*` families are. The loader in `ns_part12_gate.py`
prints a diagnostic naming the file and format if a read fails.

## Important: verify the data layout matches

The HWY repository is a Julia / Jupyter interval-arithmetic verification
project; its `data/` directory holds the precomputed profile and
eigenpair. Our Python operator realization (`src/ns_part12_gate.py`,
`src/ns_part3k.py`) expects those coefficient files at specific paths.
Upstream reorganizations are possible, so before trusting any result,
confirm the file names/layout our loaders reference still match the
current contents of the authors' `data/` directory.
`src/ns_part12_gate.py` md5-gates the data it loads and fails loudly on
a mismatch (profile-equation residual 6.2e-9; stored eigenpair residual
1.7e-5 must pass) rather than producing silent garbage — that gate is
the check to watch.

## Derived intermediates

Gram caches (~0.3–1.3 GB each) are written next to the scripts and are
fully regenerable; they are deliberately not in version control.
