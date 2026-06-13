# SPDX-License-Identifier: MIT
# Copyright (c) 2026 P. Salmond — hwy-spectral-survey (see LICENSE, NOTICE)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
Z = np.load('nsx_x5_grid.npz')
R, Re, Im = Z['R'], Z['Re'], Z['Im']
Zr = np.load('nsx_x5_real.npz')
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.0),
                             gridspec_kw=dict(width_ratios=[1.15, 1]))
a1.semilogy(Zr['lams'], Zr['r'], '-', color='#1a5fb4', lw=1.8)
a1.axvline(0.11314203, color='#c01c28', ls=':', lw=1.2)
a1.plot([0.113142], [9.3141e-3], 'o', ms=5, color='#c01c28')
a1.set_xlabel(r'$\lambda$ (real axis)')
a1.set_ylabel(r'$r(\lambda)$')
a1.set_title(r'real axis: vertex $0.113142$, depth $9.31\times 10^{-3}$',
             fontsize=9.5)
a1.grid(alpha=0.25, which='both')
ext = [Re[0]-0.025, Re[-1]+0.025, Im[0]-0.025, Im[-1]+0.025]
im = a2.imshow(np.log10(R).T, origin='lower', aspect='auto', extent=ext,
               cmap='viridis')
a2.plot([0.11314], [0.0], '*', ms=11, color='#c01c28',
        clip_on=False, zorder=5)
a2.set_xlabel(r'$\mathrm{Re}\,\lambda$')
a2.set_ylabel(r'$\mathrm{Im}\,\lambda$')
a2.set_title(r'complex window: $\log_{10} r(\lambda)$ (no off-axis minima)',
             fontsize=9.5)
fig.colorbar(im, ax=a2, shrink=0.9)
fig.tight_layout()
fig.savefig('landscape_x5_full.pdf')
print("heatmap written; grid min:", R.min(), "at Re,Im =",
      Re[np.unravel_index(R.argmin(), R.shape)[0]],
      Im[np.unravel_index(R.argmin(), R.shape)[1]])
print("monotone in Im per row:",
      all(np.all(np.diff(R[i]) > 0) for i in range(len(Re))))
