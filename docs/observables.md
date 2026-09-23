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
```
