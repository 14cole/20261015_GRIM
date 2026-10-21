# Thin dielectric strip example

Load `thin_strip.geo` with geometry units set to **meters**, solver **2D**,
frequency **1 GHz**. Suggested monostatic angles: 12, 48 and 86 degrees.
The line is 100 mm long; thickness is 0.5 mm; epsilon is 3 - j0.02, mu is 1.
Thickness is stored in meters even when coordinate units differ.

The model is a transmitting, uniform, isotropic layer in air. It is not a BoR
thin dielectric or an IBC on a dielectric body. Use the accuracy report to check
mesh convergence and compare against explicit thickness for your application.
Converged meshes do not certify the thin-layer approximation.

Production also compares the layer's local plane-wave reflection/transmission
and in-domain guided-wave propagation indices with an independent finite-thickness
slab. A failed comparison requires explicit bulk geometry. Surface-wave
sampling is checked before mesh convergence can be certified. Passing these
local checks is not a physical error bound for a finite strip, curved shell,
or resonant geometry; inspect the separate evidence in the solve report.

`tests/test_thin_sheet.py` checks complex finite-annulus references,
transparency, open-strip reciprocity, numerical routes and material round trips.
The local explicit-strip probe at these angles agreed within 0.067 dB in both
polarizations; this is one finite-bulk comparison, not a universal error bound.
