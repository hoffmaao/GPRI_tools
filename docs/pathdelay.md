# The path delay, from the pairs themselves

[`docs/atmosphere.md`](atmosphere.md) separates air from ice **spatially**: a
screen fitted on stable ground and subtracted everywhere. That needs bedrock
in the frame, and it removes only what the screen model can express.
`gpri_tools.pathdelay` separates them **temporally** instead, the way the
terrestrial-radar literature does (Wild et al., double-difference DInSAR on
Grenzgletscher). Nothing is fitted on rock.

## The double difference

Each pair's line-of-sight displacement, in the `d_j - d_i` convention
`invert_network` uses, carries the surface's motion and the air's delay in
the same difference:

    b_p = (d_j - d_i) + (a_j - a_i)          p = (i, j)

Take two pairs sharing their middle epoch, `p = (i, j)` and `q = (j, k)`, and
difference them after dividing each by its own span:

    D = b_p / dt_p - b_q / dt_q
      = -a_i / dt_p + a_j (1 / dt_p + 1 / dt_q) - a_k / dt_q

A surface moving at a steady rate contributes `v - v = 0` — exactly, at any
spacing, which is what the division by the spans buys. Equally spaced epochs
give the familiar `(-1, 2, -1) / dt` second difference. Every such triplet is
one row of `A`; `double_difference` builds it, and `invert_path_delay` solves
`A a = b` for one delay per acquisition with Tikhonov regularisation,

    a = (A^T A + lam I)^-1 A^T b

with `lam` from generalised cross-validation (`select_lambda`) unless it is
given.

## What the system cannot see

**An affine delay.** `A` annihilates `a_i = alpha + beta t_i`: a delay that
grows linearly in time is exactly what a steady rate looks like, and the
double difference was built to remove that. The null space is two
dimensional, so the answer is unique only after a representative is chosen.
`pin_affine` removes the least-squares trend in time; `pin_rate` removes
instead the trend that would change the mean apparent velocity, so that the
correction provably leaves every rate alone. The examples use `pin_rate`,
and report the trend so discarded — the size of the unobservable part, in
the units it would otherwise be mistaken for: between **−1.01 and +1.22
m/yr** across the six campaigns.

**Slow atmosphere, or unsteady flow.** Motion and delay enter identically,
so nothing separates them but the assumption that the motion is linear in
time. Everything in the phase that is not linear in time is called
atmosphere here, whatever put it there. What reaches the system at all is
set by the operator: a component of period `T` sampled at spacing `dt`
arrives weighted by `4 sin^2(pi dt / T) / dt`, which vanishes as `T` grows,
and the regularised solve returns `g^2 / (g^2 + lam)` of it.
`frequency_response` evaluates that product for one equally spaced triplet
and `system_response` measures it on the operator actually being inverted.
**Left to cross-validation it is not small on every campaign:** GCV gives
`20170713_full` a response of 0.58 at 24 h, so the correction would take more
than half of any diurnal signal with it. That is what the weight floor below
is for, and the examples apply it by default.

## Three estimators, and the split-half score

Bedrock is not needed to estimate the delay, but it is still what scores it.
`examples/baker_pathdelay.py` keeps the ladder's rule — the stable mask is
split in half, one half feeds the estimate and the other is never touched by
it — and runs three estimators so the honest number and the flattering one
sit side by side:

| estimator | what it fits | scored how |
|---|---|---|
| `scene` | one delay series for the frame, from the spatial mean of the fit-half bedrock | on the held-out half, which it never saw |
| `pixel` | one delay series per pixel | a control: one unknown per epoch against one observation per pair, so it fits the pixel's own noise |
| `smooth` | the per-pixel field, Gaussian-smoothed over (5, 25) px | the spatially coherent part of the same field |

The score is the standard deviation of the apparent LOS velocity `b_p / dt_p`
before and after the delay is removed, in m/yr. For `scene` the mask is
averaged first; for `pixel` and `smooth` each pixel is scored on its own
series and the spreads are averaged, which is why their "before" columns are
an order of magnitude larger.

## The six campaigns

Reduction in the scatter of apparent LOS velocity, per estimator, on the
held-out bedrock the estimate never saw and on the coherent ice:

| campaign | epochs | baselines (min) | lam | raised by the floor | response 2 h | response 24 h | `scene` rock | `pixel` rock | `smooth` rock | `smooth` ice | trend discarded (m/yr) |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `20170713_full` | 271 | 4.1–42.7 | 11.35 | from 0.014 | 0.966 | 0.010 | 65.6 % | 85.2 % | 3.9 % | 5.2 % | +0.03 |
| `20170803_full` | 723 | 2.0–6.3 | 14.97 | no | 0.794 | 0.002 | 41.1 % | 84.0 % | 4.4 % | 7.8 % | −1.01 |
| `20170827` | 1,335 | 2.0–19.3 | 0.698 | from 0.083 | 0.988 | 0.010 | 64.1 % | 90.4 % | 5.3 % | 9.7 % | +0.39 |
| `20170913` | 437 | 2.0 | 5.82 | no | 0.892 | 0.005 | 56.5 % | 89.8 % | 2.9 % | 16.5 % | +0.06 |
| `20180808` | 1,227 | 2.0–26.9 | 0.839 | from 0.083 | 0.981 | 0.010 | 62.4 % | 90.4 % | 5.0 % | 7.8 % | −0.69 |
| `20190719` | 1,137 | 1.2–173.6 | 2.191 | from 0.750 | 0.957 | 0.010 | 56.9 % | 88.9 % | 5.6 % | 10.6 % | +1.22 |

Held-out bedrock carries 12,369 to 23,034 pixels per campaign and the
coherent ice 25,418 to 33,652; the fit and held halves are within one pixel
of each other by construction.

Three things the table says. The `scene` estimator, fitted on bedrock the
score never uses, takes 41 to 66 % off the bedrock scatter. The `pixel`
control takes 84 to 90 % off every mask including the held-out one — it is fitting each pixel's own noise, which is what a system
with one unknown per epoch and one observation per pair does, and it is in
the table to say that such a number measures nothing. Of that per-pixel
field only the 2.9 to 5.6 % that survives spatial smoothing is coherent
across neighbouring pixels.

Mean rates are identical before and after to the printed precision in every
row, on every mask — that is `pin_rate` holding, not a result.

![path delay, 20170913](figures/28_pathdelay_20170913.png)

*Figure: `20170913`. Top left, the delay per acquisition — the `scene`
estimate over bedrock and the `smooth` field averaged over ice — against the
specific humidity at the radar epochs. Top right, the apparent LOS velocity
of the held-out bedrock before and after the correction. Bottom left, the
smoothed delay field at one epoch. Bottom right, the response of the
inversion against period, with one hour and one day marked. Local night
(00–06) is shaded.*

## Keeping it off the signal you mean to measure

The cross-validated weight is not safe by default. Across the six campaigns
GCV picks `lam` over three orders of magnitude (0.014 to 15), and with it the
share of a diurnal the inversion returns runs from 0.0002 to 0.58. On
`20170713_full`, at the top of that range, applying the correction takes
**27 %** of the ice's diurnal amplitude with it.

`lambda_for_system_response` fixes that by choosing the weight from the
response instead of from the data: it bisects until `system_response` — the
gain the filter `V diag(s^2 / (s^2 + lam)) V^T` applies to a harmonic of that
period, measured on the operator actually being inverted — is at or under a
target. `invert_path_delay(..., protect_period=1.0)` raises `lam` to that
floor, and `examples/baker_pathdelay.py` does so by default.

The closed-form `lambda_for_response` is the same calculation for one equally
spaced triplet. Do not use it as the floor: a real stack mixes baselines and
the longer ones are more sensitive at long periods, so the closed form
understates what gets through — on a 200-epoch chain of lags 1, 2, 3 it lets
past several times the 1 % it promises.

Protecting a slow period is cheap at fast ones, because the operator's gain
falls as `1 / T^2`. Holding 24 h to 1 % on `20170713_full` raises `lam` from
0.014 to 11.35 and still returns 100 % at 10 minutes, 99.5 % at 1 h and 96.6 %
at 2 h. The period to watch is the semidiurnal one, which keeps 4.5 %. With
the floor in place the campaign that lost 27 % of its diurnal loses 0.3 %.

## What it is worth on top of the ladder

Estimated on the residual the ladder leaves, and scored only on pixels that
did not feed the estimate:

| variant | held-out rock, per-pair | held-out rock, 2 h | ice diurnal | ice, 2 h |
|---|---:|---:|---:|---:|
| fitted on fit-half bedrock | −4.2 to −5.9 % | +1.1 to −5.6 % | ±0.0 % | −0.1 % |
| coherence-weighted, held-out excluded | −0.4 to −5.3 % | +0.6 to −13.9 % | −0.0 to −1.3 % | −13.9 to −27.9 % |
| ice-only fit | +235 to +340 % | +32 to +143 % | −0.0 to −1.3 % | −13.9 to −22.6 % |

(`20170713_full`, `20170803_full`, `20170913` and `20180808`, `lam` floored at
1 % of 24 h.)

Held-out bedrock improves by a few per cent, most on `20170803_full`, the day
with the most atmosphere left after the ladder. The ice's 2 h scatter drops
14 to 28 % — but a delay fitted on bedrock alone removes 0.1 % of it, and a
delay fitted on the ice alone removes all of it while making bedrock 235 to
340 % worse. A component shared between rock and ice would appear in the
bedrock-fitted estimate; none does.

Two ways to get a wrong answer here, both found by getting them wrong:
smoothing the raw per-pixel field with a plain Gaussian over the whole frame
pulls in delays fitted on incoherent pixels (that field carries 0.111 mm over
held-out bedrock against 0.008 mm when coherence-weighted, and it invents a
doubling of the 2 h velocity noise); and leaving the held-out pixels in the
weights turns a −0.7 % non-result into a −41.9 % improvement.

## Against the humidity beside the radar

`refractivity.specific_humidity` turns the ERA5 temperature, relative
humidity and surface pressure the met cache already carries at each radar
epoch into g/kg, and `humidity_gradient` differences two levels. Correlated
against the retrieved delay, epoch by epoch, the six campaigns give r =
−0.04, −0.11, −0.27, −0.27, −0.37 and −0.40 over humidities of 2.6 to 11.6
g/kg.

That comparison is limited on both sides: ERA5 is hourly at a grid point,
while the delay this method recovers is by construction the part that varies
faster than the regularisation cuts off — an hour or less on five of the six
campaigns. A station logging at the radar's own cadence is what the
comparison needs.

## Running it

```bash
python examples/baker_pathdelay.py --scene 20170913
python examples/baker_pathdelay.py --scene 20170803_full --lags 1 2 3
```

The second form forms pairs over several temporal baselines, which carries
less noise into the delay — measured on a synthetic 40-epoch chain, lags
1, 2, 3 propagate a quarter of the noise that lag 1 alone does — at the cost
of reading the stack again. Results cache to
`work/<scene>/pathdelay_u_dec16.npz` and the figure to
`docs/figures/28_pathdelay_<scene>.png`.
