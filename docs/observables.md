# Other things this data measures

The rest of this repository estimates one quantity — line-of-sight
displacement — and spends its effort separating it from the atmosphere.  The
same acquisitions carry several other measurements that cost nothing extra,
because they are made from products already on disk.  This page collects
them.  Each one is a library module, an example script, a figure and a
number; none of them reinterprets the deformation analysis.

Every script here reads the corrected series the rest of the analysis
ships: reference, per-epoch drift, per-epoch turbulence screen, then the
temporal path delay of [`gpri_tools.pathdelay`](pathdelay.md), which
`--no-path-delay` leaves in where the script takes that flag.

## The 24 h harmonic, per pixel

`baker_population.py` reads the median of the ice against the median of the
held-out bedrock; `examples/baker_harmonics.py` fits the same 24 h harmonic
to every pixel (`gpri_tools.diurnal.fit_harmonics`) and maps amplitude, hour
of peak, and amplitude over its own one-sigma.

That third quantity is new: `HarmonicFit.amplitude_sigma` propagates each
pixel's residual scatter through the least-squares covariance of the two
harmonic coefficients, so an amplitude can be read against the error on it.
On a Monte Carlo of 300 noisy fits the predicted sigma is the measured
spread to better than a percent.

![the 24 h harmonic per pixel, 20170803](figures/32_harmonics_20170803_full.png)

| campaign | ice amplitude | ice peak | rock amplitude | rock's p84 | ice above it | amplitude vs range |
|---|---:|---:|---:|---:|---:|---|
| `20170713_full` | 12.34 mm | 04.7 h | 7.43 mm | 12.87 mm | 46.8 % | r = −0.00 |
| `20170803_full` | 16.71 mm | 23.2 h | 6.94 mm | 13.37 mm | 61.6 % | r = +0.37 |
| `20170827` | 14.12 mm | 19.0 h | 8.16 mm | 14.33 mm | 48.9 % | r = +0.17 |
| `20180808` | 17.07 mm | 00.3 h | 7.50 mm | 13.81 mm | 61.5 % | r = +0.38 |
| `20190719` | 12.28 mm | 10.4 h | 7.01 mm | 12.33 mm | 49.7 % | r = +0.26 |

Amplitudes are medians over the mask; the hour is the circular mean over
pixels that clear three sigma, which on these records is 97.6 to 99.4 % of
the ice.  The last column is
`gpri_tools.diurnal.range_dependence` on the per-pixel amplitude: a diurnal
that grows with distance from the tripod is residual refractivity, and the
script prints that verdict with the number.  Note what the fourth and fifth
columns say about the per-pixel reading: the held-out bedrock, which does
not move, carries a median 24 h amplitude of 7 to 8 mm, and between 47 and
62 % of ice pixels exceed the bedrock's 84th percentile.

The per-pixel fit is therefore a weaker statement than the population
median of [`baker.md`](baker.md), not a stronger one: it measures each
pixel against its own noise, and the bedrock control says what that noise
is worth.

## Decorrelation: coherence as the measurement

Everywhere else coherence is a weight.  The long-baseline pair caches — the
ones the multi-baseline path delay needed, lags of 1 to 360 or 720 epochs —
carry a coherence per pair per pixel, so every pixel's decorrelation curve
is already on disk.  `gpri_tools.decorrelation` reduces the stack to one
coherence per baseline class (by epoch lag, which is immune to cadence
jitter) and reads off the baseline at which each pixel crosses a coherence
level.  `examples/baker_decorrelation.py` maps it.

![decorrelation, 20170913](figures/33_decorrelation_20170913_lk3x15.png)

| campaign | classes | ice: time to γ = 0.5 | crosses | rock: time to γ = 0.5 | crosses | fitted τ, ice / rock |
|---|---:|---:|---:|---:|---:|---|
| `20170803_full` | 8 | 0.76 h | 82.8 % | 4.16 h | 53.6 % | 0.84 / 2.71 h |
| `20170827` | 9 | 1.06 h | 81.4 % | 5.53 h | 50.7 % | 1.01 / 3.19 h |
| `20170913` | 8 | 4.27 h | 73.9 % | 7.83 h | 27.6 % | 2.69 / 5.46 h |
| `20180709` | 6 | 0.69 h | 74.9 % | 1.76 h | 28.4 % | 0.87 / 1.62 h |
| `20180808` | 9 | 0.97 h | 80.2 % | 5.01 h | 56.9 % | 0.89 / 1.73 h |
| `20190719` | 9 | 1.52 h | 88.7 % | 4.41 h | 71.2 % | 1.35 / 3.49 h |

The median is over the pixels that do cross within the record; the "crosses"
column says how many those are, and it belongs beside the median because a
pixel that never reaches γ = 0.5 has no crossing time and is not averaged
in.  `τ` is from `exponential_decorrelation`, the usual
`(γ0 − γ∞) exp(−t/τ) + γ∞` fitted to the mask-median curve, which is a
different estimator answering a related question: the curve's scale rather
than a level crossing.

Two limits.  A coherence estimated from `L` looks is biased high, so the
long-baseline end is a ceiling, not a measurement; the bias depends only on
`L`, so it moves every class the same way and the shape survives.  And a
crossing time is bounded by the record: `20170913` spans 14.5 h and its
longest class is 12 h, so nothing slower than that is measurable in it.

## Turbulence: what the screen is removing

The ladder convolves at `sigma = (5, 25)` pixels, which is a claim about the
distance over which the atmosphere stays coherent.
`gpri_tools.turbulence.structure_function` measures it:
`D(r) = <[φ(x+r) − φ(x)]²>` over pixel pairs binned by ground separation, on
bedrock, per epoch.  A structure function needs no mean, so the arbitrary
constant and ramp a bedrock-fitted screen carries do not enter it.

`examples/baker_turbulence.py` measures the epoch-to-epoch increment by
default — what the atmosphere did over one cadence — because the integrated
series' structure function grows with time from the reference epoch, which
is a property of the integration and not of the air.

![the structure function of the screens, 20170803](figures/34_turbulence_20170803_full.png)

| campaign | exponent (p16–p84) | D at 1 km | rms difference at 1 km |
|---|---|---:|---:|
| `20170713_full` | +0.10 (+0.06 to +0.17) | 8.62 mm² | 2.94 mm |
| `20170803_full` | +0.10 (+0.07 to +0.16) | 2.93 mm² | 1.71 mm |
| `20170827` | +0.15 (+0.08 to +0.25) | 3.65 mm² | 1.91 mm |
| `20170913` | +0.10 (+0.08 to +0.12) | 1.57 mm² | 1.25 mm |
| `20180709` | +0.17 (+0.12 to +0.27) | 3.86 mm² | 1.96 mm |
| `20180808` | +0.10 (+0.05 to +0.16) | 4.68 mm² | 2.16 mm |
| `20190719` | +0.14 (+0.08 to +0.21) | 3.49 mm² | 1.87 mm |

Kolmogorov phase turbulence gives an exponent between 2/3 and 5/3 depending
on the regime, and the figure shades that band.  The measured exponents are
+0.10 to +0.17 over separations of tens of metres to a few kilometres: over
one cadence, at these separations, the field the screen sees is nearly flat
in separation.  The amplitude column is the size of what is there —
1.3 to 2.9 mm rms between two bedrock pixels a kilometre apart, one cadence
apart in time.

The script prints the ladder's own sigma converted to metres on the same
geometry (89 × 296 m on `20170713_full` at dec 16), so the smoothing scale
and the measured separations can be read against each other.

## The two antennas as an interferometer

The GPRI-II receives on two antennas 25 cm apart on one mast, sampled in the
same sweep, so `s_upper * conj(s_lower)` at one epoch has no temporal
decorrelation, no deformation and no atmospheric delay beyond the difference
over 25 cm.  What is left is geometry — the path-length difference set by
each target's elevation angle — which makes it a topographic measurement the
radar makes of itself.  `examples/baker_antenna_interferogram.py` forms it
and compares it against the DEM `gpri_tools.heading.target_heights` reads.

![the cross-antenna interferogram, 20170913](figures/35_antenna_ifg_20170913_lk3x15.png)

Two things had to be handled before the comparison meant anything.  The two
receive chains carry a relative phase that is constant across the scene and
changes sweep to sweep; averaging epochs without removing it cancels the
geometry instead of the noise, so each epoch's scene-constant phase is taken
out before the epochs are stacked.  And the fit of the measured phase against
the DEM's `sin(theta)` is **aliased**: with `2 pi B / lambda` of order ninety
radians, slopes differing by about 0.7 fit nearly as well, so the script
scores the slopes the geometry allows (one-way, two-way, half), refines
locally, and prints the unconstrained scan beside them rather than trusting
it.

| campaign | resultant at k = 1 | refined slope | resultant | best registration | height agreement |
|---|---:|---:|---:|---:|---:|
| `20170713_full` | 0.160 | +1.21 | 0.411 | 0.434 | 65 m |
| `20170803_full` | 0.172 | +1.21 | 0.426 | 0.442 | 62 m |
| `20170827` | 0.013 | +0.81 | 0.021 | 0.028 | 199 m |
| **`20170913`** | **0.897** | **+0.98** | **0.906** | 0.906 | **26 m** |
| `20180808` | 0.023 | +2.23 | 0.051 | 0.057 | 63 m |
| `20190719` | 0.014 | +0.81 | 0.018 | 0.024 | 203 m |

On `20170913` the interferogram reads topography and says so unambiguously:
the one-way slope wins at 0.897 against 0.048 for two-way and 0.149 for half,
the refinement lands at 0.98 of the one-way prediction, and the residual
against the DEM is 26 m of height at the median range of 5.2 km, where a
cycle is 369 m.  The measured and predicted fringe patterns in the figure are
the same picture.

On the other five it does not, and three explanations were checked and ruled
out: the two antennas' epoch lists align exactly by index on every campaign
(437/437, 1335/1335 and so on); the per-epoch cross-antenna coherence is the
same everywhere (0.174 to 0.181 median, epoch-to-epoch consistency 0.279 to
0.286), so the interferograms are of equal quality; and scanning the DEM's
registration over ±12 azimuth and ±4 range pixels moves the agreement by at
most 0.02.  What separates `20170913` from the rest is therefore still open.

## Speckle tracking, and what it cannot see here

`gpri_tools.tracking.patch_offsets` cross-correlates intensity patches, which
measures displacement with no phase ambiguity and no coherence requirement,
at a resolution set by the cell size.  `examples/baker_tracking.py` runs it
between two epochs and puts the answer beside the phase's over the same
interval.

The result is a null with a number on it.  Over 2 h on `20170913`, held-out
bedrock gives a range offset of −0.019 ± 0.054 samples — 41 mm in line of
sight, against the phase's 1.94 mm over the same interval; on
`20170803_full`, −0.002 ± 0.100 samples, 75 mm against 10.68 mm.  Only about
a third of patches clear a correlation of 0.3 at all, the median peak
correlation is 0.02, and **no ice patch ever clears it**: the pattern inside
a patch is speckle, which does not repeat over hours, rather than terrain
texture that would.

So tracking on this instrument at these separations cannot see the ice, and
on rock it is some forty times coarser than the phase.  Read the other way,
the noise floor says how fast a target would have to move before tracking
could measure it at all — of order a metre a day in line of sight at three
sigma over a 2 h pair — which is the number to have before reaching for
tracking on faster ice than Baker's.  Multilooking the intensity first
(`--looks`) raises the median correlation to 0.09 and makes the offsets
noisier, so it is not the default.

## Persistent scatterers in the far field

`gpri_tools.psinterp` implements Chen, Zebker & Knight's PS-interpolation
unwrapping and no Baker script had used it.  The place it belongs is the
weak spot [`baker.md`](baker.md) documents: beyond 7 km, where 46 % of the
ice sits against 308 of the 7,697 stable pixels, the rock-fitted screens are
extrapolating.  `examples/baker_ps.py` selects scatterers by amplitude
dispersion over the record, unwraps a long-baseline interferogram at them
along a minimum spanning tree in **ground metres** (the GPRI's pixel spacing
is anisotropic enough that a tree built in pixels picks the wrong
neighbours), and interpolates back onto the grid.

![persistent scatterers, 20170913](figures/37_ps_20170913_6h.png)

On `20170913` at 3 × 15 looks, 40,000 pixels pass a dispersion of 0.25 —
18 % of the grid, 3,185 of them on ice and 2,644 on bedrock — and their
density rises with range rather than falling:

| range (km) | 0–1 | 2–3 | 4–5 | 5–6 | 6–7 | 7–8 | 9–10 | 12–13 | 16–17 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PS share | 1.7 % | 11.9 % | 17.1 % | 26.4 % | 28.7 % | 24.9 % | 19.1 % | 18.4 % | 19.1 % |
| PS on ice | 0 | 0 | 23 | 800 | 615 | 999 | 54 | 0 | 0 |

So the far field the screens cannot reach is not empty of usable targets:
at 7–8 km, where the modelled stratification residual is 70 % of what it
started as, there are 999 persistent scatterers on the ice itself.

Unwrapping the 6 h interferogram at those targets moved 86 % of ice pixels
by a whole cycle or more (up to five cycles) and 47 % of bedrock pixels (up
to seven), with no unresolved targets and a suspect fraction — pixels whose
residual came back near π, where the subtract-wrap-add step aliased — of
8.7 % on ice and 4.7 % on bedrock.  That suspect map is the honest failure
map, not a quality score.

## Phase linking, and the quality number it brings

`gpri_tools.covariance` and `gpri_tools.phaselink` had likewise never been
run on this data.  `examples/baker_phaselink.py` builds the N × N sample
coherence matrix per pixel and factors it for the per-epoch phases that
explain every pair at once, rather than the chain's consecutive differences.

The configuration decides the answer, and the script makes that visible with
`--spread`:

| mini-stack | temporal coherence, ice / rock | linked sd, ice | chain sd, ice | linked − chain |
|---|---|---:|---:|---:|
| 24 consecutive (48 min) | 0.999 / 0.995 | 4.07 mm | 4.01 mm | 1.40 mm |
| 24 spread over 13.8 h | 0.038 / 0.048 | 4.98 mm | 27.70 mm | 28.12 mm |

Linked consecutively the rank-one model is essentially exact and the series
agrees with the chain to 1.4 mm; spread across the record, the pairs have
decorrelated (the decorrelation section above says ice reaches γ = 0.5 in
4.3 h on this campaign), the model fits nothing, and the "linked" series is
flat where the chain has 27.7 mm of motion.

What linking adds is `temporal_coherence`, a per-pixel number the chain
cannot produce: on the consecutive mini-stack 96.3 % of ice pixels and
61.1 % of held-out bedrock reach 0.6.  The chain cannot tell a pixel whose
steps are mutually consistent from one whose steps merely integrate.

## Velocity gradients

The spatial derivative of the per-pixel rate is what a strain rate is made
of.  `examples/baker_strain.py` fits a plane to the line-of-sight rate over
a 200 m ground neighbourhood at every pixel and maps the gradient, taking
the rates from `baker_harmonics.py`'s cache where one exists.

![velocity gradients, 20170827](figures/39_strain_20170827.png)

On `20170827`: the ice's rate is +13.7 m/yr (p16–p84 −7.3 to +41.6) and its
gradient 64.1 m/yr per km (p16–p84 28.8 to 118.4); held-out bedrock, which
does not move, gives +0.2 m/yr and a gradient of 37.9 m/yr per km (19.4 to
68.3).  **The ice's gradient is 1.7 times the bedrock's**, and that ratio is
what the map is worth: differentiating a rate differentiates its noise, and
the bedrock row says how much of the ice's structure is that.  Across the
five campaigns that carry a per-pixel rate the ratio runs 1.3, 1.7, 2.6, 3.1
and 3.6 (`20170713_full`, `20170827`, `20180808`, `20190719`,
`20170803_full`), so how much the gradient map is worth is a property of the
campaign, not of the method.

Two limits the script prints rather than hides: one look direction gives one
component of a three-dimensional velocity field, so this is the gradient of
the line-of-sight rate and not a strain-rate tensor; turning it into one
needs a flow direction and a depth assumption this data cannot supply.

## Reproduce

```bash
# the 24 h harmonic per pixel, for every campaign that spans a cycle
for s in 20170713_full 20170803_full 20170827 20180808 20190719; do
  python examples/baker_harmonics.py --scene $s --decimate 16 --rgi
done
# coherence against temporal baseline, from the long-baseline pair caches
python examples/baker_decorrelation.py --scene 20170913 --lags 1 2 3 30 60 90 180 360
python examples/baker_decorrelation.py --scene 20170827 --lags 1 2 3 30 60 90 180 360 720
# the structure function of what the turbulence screen removes
for s in $CAMPAIGNS 20180709; do
  python examples/baker_turbulence.py --scene $s --rgi
done
# the two antennas as an interferometer, against the DEM
for s in $CAMPAIGNS; do
  python examples/baker_antenna_interferogram.py --scene $s --epochs 16
done
# speckle tracking beside the phase, over a two-hour pair
python examples/baker_tracking.py --scene 20170913 --hours 2 --rgi
# persistent scatterers, and unwrapping a six-hour interferogram at them
python examples/baker_ps.py --scene 20170913 --hours 6
# phase linking a consecutive mini-stack, and the same epochs spread out
python examples/baker_phaselink.py --scene 20170913 --epochs 24
python examples/baker_phaselink.py --scene 20170913 --epochs 24 --spread even
# the gradient of the per-pixel rate (needs baker_harmonics.py first)
for s in $CAMPAIGNS; do python examples/baker_strain.py --scene $s --rgi; done
```
