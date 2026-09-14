# Atmospheric correction: what actually works, measured

Built and validated 2026-08-30. Reproduce with:

```bash
python examples/baker_aps.py --scene 20170713 --decimate 4  --sigma 5 40 --screens-on-bedrock
python examples/baker_aps.py --scene 20170803 --decimate 16 --sigma 5 25
python examples/baker_closure.py
```

## The ladder, and how it is scored

Four correction stages, each including the ones above it, scored on the RMS of
the displacement time series over **held-out bedrock**: the stable mask is
split in half, one half feeds every correction, the other half only ever
scores them. Bedrock is not moving, so whatever remains there is error, and
no correction ever saw the scoring pixels.

| stage | correction | machinery |
|---|---|---|
| A | reference only | per-epoch constant tied to bedrock |
| B | + per-pair screens | matched-filter ramp + robust linear fit per interferogram |
| C | + drift removal | `gpri_tools.aps.epoch_screen_correction` — screen refitted per epoch on the integrated displacement over bedrock |
| D | + turbulence | `gpri_tools.aps.turbulence_screen` — normalised convolution of each epoch's residual over bedrock |

## Results

**20170713** (246 pairs, 21.8 h, dec 4, screens fitted on bedrock):

| stage | held-out bedrock RMS | vs A |
|---|---:|---:|
| A reference only | 25.93 mm | 100.0 % |
| B + pair screens | 26.25 mm | 101.2 % |
| C + drift removal | 26.01 mm | 100.3 % |
| D + turbulence | **20.94 mm** | **80.8 %** |

**20170803** (722 pairs, 24.2 h, dec 16, screens on everything above coherence 0.4):

| stage | held-out bedrock RMS | vs A |
|---|---:|---:|
| A reference only | 47.08 mm | 100.0 % |
| B + pair screens | 49.22 mm | 104.5 % |
| C + drift removal | 46.83 mm | 99.5 % |
| D + turbulence | **30.05 mm** | **63.8 %** |

The accumulated ramp drift stage C removed was 15.95 N-units peak-to-peak over
the 722 pairs.

## Revision after the RGI audit (2026-08-31, rescored 2026-09-02)

The tables above were scored against a stable mask chosen by coherence alone —
and the Randolph Glacier Inventory audit (`examples/baker_rgi.py`) later
showed **62.8 % of that mask was on glacier**. Rescoring the 20170803 ladder
with a true-rock reference and null (`--rgi`), on the campaign's measured
scan heading (107.4°; the 105° the first RGI pass used put the mask 2.4° off
its ground):

| stage | ice-contaminated ref | RGI-corrected ref | RGI, lower antenna | 20170827 upper | 20170827 lower |
|---|---:|---:|---:|---:|---:|
| A reference only | 47.08 mm | **27.14 mm** | 28.64 mm | 35.80 mm | 36.41 mm |
| B + pair screens | 49.22 mm | 34.70 mm | 37.16 mm | 46.32 mm | 45.80 mm |
| C + drift removal | 46.83 mm | 32.81 mm | 33.76 mm | 46.37 mm | 45.48 mm |
| D + turbulence | 30.05 mm | 28.41 mm | 29.65 mm | 39.44 mm | 38.47 mm |

The third column is the same day seen by the GPRI's second (lower) receive
antenna, formed from its SLCs with `gpri_tools.stack.SlcPairStack` and run through
the identical ladder (`--antenna lower`). It replicates the upper antenna's
table stage for stage — same shape, same verdict on the per-pair screens —
with a slightly larger held-out reference (4,551 px against 3,816; the
lower antenna's coherence is marginally higher). The validated recipe
without stage B (reference + drift removal + turbulence) scores 23.3 mm on
the upper antenna's held-out rock and 23.5 mm on the lower's; of that,
`examples/baker_antennas.py` measures 16.7 mm as single-antenna noise (RMS
of upper − lower over √2) and 16.2 mm as common-mode error the two channels
share.

The last two columns are the two-day campaign, `20170827` (44.9 h, 1334
pairs, both antennas, heading 100.1°), scored the same way. The shape
repeats a third and fourth time — B worse than A, D recovering most but not
all of it — on a held-out rock set of 3,148 / 3,538 px (the campaign's
coherence is lower, median 0.28 at 5×5 looks). Stage A at 35.8 mm over
44.9 h against 27.1 mm over 24.2 h is the √t growth of single-look noise:
27.1 × √(44.9 / 24.2) = 37.0 mm predicted, so the longer record adds no
atmospheric error beyond what the extra epochs integrate.

The five campaigns processed at the time were each scored the same way, on
a true-rock reference at its measured heading, both antennas
(`20170713_full` is the July archive refocused to its full 23.9 h;
`20170913` and `20180709` are the two sub-cycle campaigns, the latter
co-registered across a 5.1° tripod drift):

| campaign | span | held-out px | A | B | C | D | D / A |
|---|---:|---:|---:|---:|---:|---:|---:|
| `20170713_full` upper | 23.9 h | 3,022 | 21.59 mm | 24.00 | 23.98 | 22.22 | 102.9 % |
| `20170713_full` lower | 23.9 h | 3,641 | 21.19 mm | 24.17 | 24.13 | 22.28 | 105.1 % |
| `20170803` upper | 24.2 h | 3,816 | 27.14 mm | 34.70 | 32.81 | 28.41 | 104.7 % |
| `20170803` lower | 24.2 h | 4,551 | 28.64 mm | 37.16 | 33.76 | 29.65 | 103.5 % |
| `20170827` upper | 44.9 h | 3,148 | 35.80 mm | 46.32 | 46.37 | 39.44 | 110.2 % |
| `20170827` lower | 44.9 h | 3,538 | 36.41 mm | 45.80 | 45.48 | 38.47 | 105.7 % |
| `20170913` upper | 14.5 h | 3,751 | **9.06 mm** | 9.70 | 9.70 | 9.63 | 106.3 % |
| `20170913` lower | 14.5 h | 4,186 | 10.97 mm | 11.35 | 11.35 | 11.29 | 103.0 % |
| `20180709` upper | 6.9 h | 3,610 | 18.09 mm | 22.60 | 22.59 | 18.99 | 104.9 % |
| `20180709` lower | 6.9 h | 4,081 | 18.73 mm | 22.43 | 22.41 | 19.52 | 104.2 % |

On true rock at the right heading the ladder gains nothing on any of them:
stage D is 103–110 % of plain referencing on all ten rows, and the per-pair
screens cost 3–30 %. (The 88–90 % July showed when it was first rescored
was scored with a mask drawn at 105° instead of its measured 111.4° — 6.4°,
560 m at 5 km — so its "held-out rock" was partly the wrong pixels; the
gain went with the mask.) The two August campaigns are consistent with one
noise level growing as √t (27.1 over 24.2 h; 35.8 over 44.9, where 27.1 ×
√(44.9 / 24.2) = 37.0). The other three are not on that curve: July sits
lower (21.6 over 23.9 h), the 2018 morning higher (18.1 over 6.9 h against
14.5 predicted), and mid-September is the quietest atmosphere of the five
by a factor of two — 9.1 mm over 14.5 h against 21 predicted, with a
common-mode floor between the antennas of 2.6 mm against 10–16 mm on the
summer days.

The four scenes processed since — `20170803_full`, `20180808`, `20190719`
and `20160826_full` — are scored on held-out rock the same way in the
[`baker.md`](baker.md)'s ["Eight campaigns on one clock"](baker.md#eight-campaigns-on-one-clock)
table, and do not change the verdict: stage D is 99–105 % of plain
referencing on all four. That table carries stage A and stage D of the
upper antenna only; the intermediate stages and the lower-antenna rows of
those four stay in each scene's `work/<scene>/logs/aps_upper` and
`aps_lower`, which the repository does not carry, so the table above stops
at the five scored here. The one exception found anywhere is `20190719`'s
lower antenna, where stage D is 83 % of A — the only row in nine scenes on
which the turbulence screen pays for itself.

Two of the conclusions below need correcting in the light of the audit:

- **Referencing to actual rock is worth more than every correction
  combined.** Plain stage A on a true-rock reference (27.1 mm) beats the
  old fully-corrected stage D (30.1 mm).
- **The turbulence screen's 36 % gain was an artefact of the moving
  reference.** Fitted on a mask that was mostly ice, it partly learned
  and subtracted spatially smooth *glacier motion* — which lowered "bedrock"
  RMS only because the held-out "bedrock" was the same moving ice. On true
  rock it does not gain at all, further throttled because the genuine rock
  area (3,816 px at this decimation) supports the kernel over only 6.3 % of
  the grid. The method stands; the measured gain did not.

What survives unchanged: per-pair screens still do not pay for themselves
(B > A in every configuration), the drift argument still holds, and the
√t single-look noise floor is unchanged. The durable lesson is sharper,
though: **get the reference right before correcting anything** — coherence
is not stationarity, no atmospheric model can fix a reference that moves,
and a mask is only as good as the heading it is drawn at.

## What the numbers say

**1. Per-pair parametric screens do not pay for themselves here.** On
20170713, stage B is *worse* than doing nothing beyond referencing — 103.5 %
of A with the original configuration (screens fitted on everything above
coherence 0.4), 101.2 % when fitted on bedrock only. The per-pair fits inject
about as much ramp noise as they remove atmosphere, and integrating 246 of
them turns that noise into a random walk: the accumulated ramp drift stage C
removes was 9.45 N-units peak-to-peak with unrestricted fits, 5.25 with
bedrock-only fits — meaning roughly **half of the "atmospheric drift" was
manufactured by the correction itself**. Stage C exists to make stage B
harmless, and it does (C ≈ A in both configurations).

This does not make the per-pair screens useless — they are what keeps each
individual *interferogram* interpretable, and the per-epoch ΔN series they
carry is the physical check against met data. It means they should not be
trusted to improve the *integrated time series*, and now there is a number
saying so.

**2. The turbulence screen is the workhorse.** The only stage that clearly
helps, on both days: 19 % RMS reduction on 20170713 and **36 %** on 20170803, the day with the stronger atmosphere. It is also the
only stage that estimates spatially-structured error non-parametrically —
which is exactly the part the earlier diurnal analysis flagged, when 41 % of
residual variance was still explained by refractivity after linear screens.

Its honest limits: it is supported only where bedrock lies within the kernel
(16.4 % of the 20170713 grid at σ = (5, 40) px); everywhere else the screen is
zero and stage D degenerates to stage C. Widening the kernel or lowering the
stable-ground threshold extends coverage at the price of a smoother, weaker
correction.

**3. What remains after D is mostly not atmosphere.** The error-growth panel
of `12_aps_*.png` shows RMS rising as √t — a per-pixel random walk from
single-look phase noise integrating over hundreds of pairs. That component is
spatially uncorrelated, so no atmospheric model can or should remove it; it
averages down as √N under spatial averaging or multilooking, which is the
correct next lever (and what `gpri_tools.phaselink` is for). Treat ~20 mm per pixel
at 22 h as the single-look noise floor of these stacks, not as an atmospheric
residual.

**4. Closure phase is now measured on real data.**
The merged single-reference + chain networks of 20160826 give 25 triangles.
On 1-look pixels closure is identically zero — an algebraic fact worth knowing
before anyone runs a closure analysis on unlooked data (it also end-to-end
validates the pair bookkeeping). After a 3×15 boxcar, closure RMS is 0.89 rad
on the best quartile of pixels; the fitted bias grows from ~0 at 5 min to
~0.08 rad (~0.1 mm LOS) at 3 h — the short-baseline fading shape — and the
correction removes 36 % of the closure RMS. At ~0.1 mm it is far below the
atmospheric error at Baker, but on the 45-hour 20170827 campaign (whose
`itab` closes natively) it accumulates over an order of magnitude more pairs.

## The stratified term

A height-dependent (stratified) correction, standard in spaceborne InSAR, is
unidentifiable for a GPRI without a DEM: every pixel shares one antenna
elevation angle, so beam height is exactly `alt + r·sin(elev)` — perfectly
linear in slant range, and absorbed indistinguishably by the uniform-mixing
ramp the matched filter already fits. Separating them requires per-pixel
terrain height, which the DEM behind `gpri heading` supplies
(`gpri_tools.heading.target_heights`).

It is fitted as a covariate rather than as a stage of its own:
`epoch_screen_correction(..., covariates={"height": z_px})` appends the
centred height column to each epoch's design matrix, and
`examples/baker_population.py --height-screen` runs the ladder that way,
writing its products beside the standard ones. What that changed on the
Baker data, and what it did not, is in [`baker.md`](baker.md) ("The weather,
and what the ice does with it"). Where the fit has nothing to constrain it
the old caveat stands unaltered: rock sitting at the rock's heights and
ranges cannot see a term that shows only on the higher, farther ice.

## What the corrected bedrock still does together

Stages C and D, fitted on one half of the bedrock, leave the other half with
a mean series that is not flat: a 24 h harmonic of 0.68 mm peak to peak on
`20170803_full` (r² 0.81 with an offset and a trend fitted beside it),
0.36 mm on `20180808` (r² 0.41) and 0.69 mm on `20190719` (r² 0.92), on
held-out means whose sd is 0.38, 0.23 and 0.51 mm. `gpri_tools.modes` asks
how that residual is organised, and `examples/baker_modes.py` runs it:
the post-ladder residual over the fitted half, each pixel centred and
scaled by the square root of its mean coherence, is decomposed by SVD into
**temporal modes** (epoch series in mm at unit loading) and **loadings**
(how much of each series a pixel carries, scaled to RMS 1 over the fitted
pixels weighted — the plain RMS is 1 only at equal weights, and 1.17–1.19
here),
against a null made by shuffling each pixel's epochs, which keeps every
pixel's variance and destroys the temporal structure. The held-out half,
which never enters the decomposition, is then scored two ways:

- **interpolated** — each mode's loading is smoothed from the fitted pixels
  onto the held-out ones with `turbulence_screen`, the ladder's own
  normalised convolution at the ladder's σ = (5, 25), and the correction
  that predicts is subtracted;
- **self-fit** — each held-out pixel is regressed on the mode series
  itself, which is the most the modes could remove if every pixel's loading
  were measured rather than interpolated.

Reproduce with `python examples/baker_modes.py --scene <campaign>`; the
figures are `figures/30_modes_<campaign>.png`.

**The modes.** Singular values over the shuffled null, the fraction of the
weighted variance each mode carries, and the 24 h swing of the mode series
(peak to peak, r² of an offset + trend + 24 h fit):

| campaign | fitted px | mode | s / null | explained | 24 h swing of series |
|---|---:|---:|---:|---:|---:|
| `20170803_full` | 17,364 | 1 | 14.3 | 56.6 % | 4.4 mm (r² 1.00) |
| | | 2 | 8.6 | 18.7 % | 18.1 mm (0.97) |
| | | 3 | 5.2 | 6.6 % | 16.8 mm (0.98) |
| | | 4–10 | 4.1 → 1.6 | 4.2 → 0.6 % | ≤ 1.0 mm (≤ 0.02) |
| `20180808` | 23,034 | 1 | 19.6 | 62.8 % | 7.8 mm (0.99) |
| | | 2 | 9.5 | 14.1 % | 12.1 mm (0.26) |
| | | 3 | 6.7 | 7.0 % | 14.9 mm (0.90) |
| | | 4 | 4.7 | 3.5 % | 9.7 mm (0.67) |
| | | 5–10 | 3.8 → 1.9 | 2.2 → 0.6 % | ≤ 2.4 mm (≤ 0.08) |
| `20190719` | 17,265 | 1 | 18.0 | 64.9 % | 5.2 mm (0.97) |
| | | 2 | 8.0 | 12.3 % | 7.6 mm (0.14) |
| | | 3 | 6.4 | 7.7 % | 15.0 mm (0.85) |
| | | 4 | 4.2 | 3.3 % | 9.5 mm (0.76) |
| | | 5–10 | 3.5 → 1.7 | 2.3 → 0.5 % | ≤ 2.3 mm (≤ 0.08) |

The tenth mode still sits 1.6–1.9 times above its null. The loadings of the
first three modes have sd 1.17–1.19, a mean within ±0.04 and a correlation
with slant range of |r| ≤ 0.03 on every campaign.

**Whether the loading transfers.** For the mode whose series carries the
largest 24 h swing, the loading each held-out pixel gets by regression
against the loading interpolated to it from its fitted neighbours
correlates at r = 0.09 (`20170803_full`, mode 2), 0.12 (`20180808`, mode 3)
and 0.10 (`20190719`, mode 3); panel (c) of the figures is that scatter.
The scores, as changes from the ladder alone (held sd and held px sd in mm;
24 h swings peak to peak; px 2 h is the median over pixels of the sd of a
pixel's 2 h velocity, mm/hr; the ice columns are the coherent ice's mean
series):

| campaign | variant | held sd | held 24 h | held px sd | held px 2 h | ice 24 h | ice rate |
|---|---|---:|---:|---:|---:|---:|---:|
| `20170803_full` | ladder | 0.384 | 0.675 | 17.96 | 6.05 | 21.99 | 23.20 m/yr |
| | + modes k=3 | −16.9 % | −15.0 % | −0.3 % | −0.0 % | −0.1 % | +0.02 |
| | + modes k=10 | −18.2 % | −13.9 % | −0.3 % | −0.0 % | −0.1 % | +0.02 |
| | self-fit k=3 | −63.9 % | −94.1 % | −57.4 % | −9.4 % | — | — |
| | self-fit k=10 | −82.6 % | −98.9 % | −75.8 % | −41.8 % | — | — |
| `20180808` | ladder | 0.227 | 0.359 | 24.96 | 6.36 | 21.01 | 25.92 m/yr |
| | + modes k=3 | +1.7 % | −4.3 % | −0.3 % | −0.0 % | +0.3 % | +0.04 |
| | + modes k=10 | +0.6 % | +1.3 % | −0.3 % | −0.2 % | +0.3 % | +0.04 |
| | self-fit k=3 | −44.2 % | −75.3 % | −60.1 % | −6.2 % | — | — |
| | self-fit k=10 | −71.8 % | −98.5 % | −76.5 % | −25.0 % | — | — |
| `20190719` | ladder | 0.513 | 0.688 | 24.12 | 6.40 | 6.31 | 19.62 m/yr |
| | + modes k=3 | −11.6 % | −15.8 % | −0.3 % | −0.0 % | −0.4 % | +0.03 |
| | + modes k=10 | −12.1 % | −17.7 % | −0.3 % | −0.2 % | −0.4 % | +0.03 |
| | self-fit k=3 | −75.3 % | −82.7 % | −60.9 % | −6.7 % | — | — |
| | self-fit k=10 | −86.1 % | −97.8 % | −77.5 % | −25.4 % | — | — |

The self-fit rows leave the ice alone by construction. Mode 1 on its own
(k = 1, not tabulated) changes the held-out 24 h swing by +0.2 to +1.6 %
interpolated and +0.2 to +18.2 % self-fitted, while taking 34–40 % off the
per-pixel sd self-fitted; its series is the one with r² ≥ 0.97 against the
trend-and-harmonic model on every campaign.

Narrower loading screens (`--loading-sigma 1 5`, the ladder kept at
σ = (5, 25)) raise the transfer correlation to r = 0.24, 0.19 and 0.22 and
at k = 3 take −11.1 %, −20.6 % and −11.7 % off the held-out 24 h swing,
−22.6 %, −7.4 % and −16.2 % off the held-out sd and −0.9 %, −0.9 % and
−1.3 % off the per-pixel sd; a kernel that narrow has no support on the
ice, and the ice columns do not move.

In one line: two or three modes carry the held-out mean's 24 h harmonic —
each held-out pixel regressed on them loses 75–94 % of it at k = 3 — and
the interpolated loadings change it by +1.3 % to −20.6 % and the
per-pixel sd by −0.3 % to −1.3 %, with r = 0.09–0.24 between a held-out
pixel's own loading and the one its fitted neighbours give it.

## Recommended pipeline

```python
from gpri_tools import aps, atmosphere
from gpri_tools.timeseries import los_displacement

# per-pair screens: keep them for per-interferogram products and the dN series,
# fit them on bedrock, and do not expect them to improve the integrated series
screens = [atmosphere.fit_screen(ph, slant_range=r, weights=w_bedrock,
                                 model="linear", wavelength=lam)
           for ph, w_bedrock in pairs]

# integrate, then let the displacement-domain corrections do the real work
d, coeffs = aps.epoch_screen_correction(displacement, fit_mask, r, "linear")
for k in range(d.shape[0]):
    scr, q = aps.turbulence_screen(d[k], fit_mask, sigma=(5, 40),
                                   weights=mean_cc, wrapped=False)
    d[k] -= scr
```

Hold pixels out of `fit_mask` before testing anything on stable ground.
