# Data (not redistributed)

This pipeline reads the public Hou-Wang-Yang repository accompanying
arXiv:2509.25116. That repository carries NO license file, so its
contents are not redistributed here. Fetch it yourself:

1. Download the repository archive
   `3d-navier-stokes-nonuniqueness-main.zip` from the authors' public
   hosting (linked from arXiv:2509.25116).
2. Unzip into this directory, so that
   `data/3d-navier-stokes-nonuniqueness-main/data/` exists, or set
   `HWY_REPO=/path/to/3d-navier-stokes-nonuniqueness-main`.
3. Integrity: the campaign's md5 gate for the archive is checked by
   `src/ns_part12_gate.py` before any computation.

Derived intermediates (Gram caches, ~0.3-1.3 GB each) are written next
to the scripts and are fully regenerable; they are deliberately not in
version control.
