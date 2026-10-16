# baker — the bed and the effective pressure of the north-side glaciers

An inverse problem built on the joint path / ice-motion inversion of
`gpri_tools.jointinv` (`examples/baker_joint.py`, `docs/joint.md`).  It
assimilates

* every radar campaign's line-of-sight strain rates between neighbouring
  200 m ice cells, at six times of day (or the mean rate alone, for the two
  campaigns shorter than a day), and
* optionally, ITS_LIVE's annual surface velocities (v2.1, 120 m,
  2017–2019) on the glaciers (`--years` with no years leaves them out),

each data set over its own surface: the 2015 lidar plus that campaign's
height change measured by the radar's two receive antennas
(`baker/radar_dem.py`, `baker/radar_joint.py`).  It estimates

* **one** ice thickness, and so one bed, shared by every glacier, campaign
  and year,
* **one** stationary effective pressure `N_bar = exp(q_bar)` under every data
  set (`--friction shared`, the default), and
* **for each** campaign and ITS_LIVE year its departure from it: a mean
  `q0` and, for the campaigns with 24 h terms, a 24 h harmonic in `log N`;
  `N(t) = exp(q_bar + q0 + qc cos wt + qs sin wt)`.  With `--friction
  separate` there is no `q_bar` and each data set has its own whole `log N`.

## Model

* **Momentum balance**: icepack2's dual (mixed) shallow-stream equations,
  with velocity, membrane stress and basal stress as unknowns.  That form is
  solvable where the thickness is zero, so the domain is a rectangle around
  the cells and ice-free ground is `H = 0` (`baker/physics.py`).
* **Flow law**: Glen's law, `n = 3`, rate factor of temperate ice (273.15 K).
* **Sliding law**: regularized Coulomb (Joughin et al. 2019),
  `|tau| = mu N (|u_b| / (|u_b| + u_0))^(1/m)`, `m = 3`, `u_0 = 300 m/yr`,
  `mu = 0.5`.  Only `mu N` enters, so `N` is read through the fixed `mu`.
  Internal deformation is added through the basal term, as in icepack2.
* **Observation**: the surface velocity (sliding plus the deformation
  profile's surface value) projected on each cell's unit vector toward the
  radar, with the vertical velocity of flow parallel to each campaign's own
  surface `s_k = s_2015 + ds_k` (see **Surfaces**); the strain rate of a
  pair is the difference over the separation (`baker/observe.py`, with an
  exact adjoint for the pair misfit).
* **ITS_LIVE**: the horizontal surface velocity at each pixel on the ice;
  each component's standard deviation is the mosaic's error combined with
  the RMS of the ice-free pixels around it (`baker/data.py`, `its_live`).
* **Surfaces**: data set `k` sees `s_k = s_2015 + ds_k` and
  `H_k = max(H_2015 + ds_k, 0)` over the one bed.  `ds_k` is the radar's own
  height change (`--surface radar`, from `--radar-dir`), the
  ASTER/ICESat-2/GEDI rate times the time since 2015 (`--surface dhdt`,
  `baker/surface.py`), or zero (`--surface lidar`).
* **Radar surfaces**: the phase between the two receive antennas at each
  epoch, stacked over 96 epochs and fitted on ice-free ground above 1700 m
  to the lidar, with each campaign's effective baseline, phase ramp, `1/R`
  term, constant and lidar shift (`baker/radar_dem.py`, which also caches
  the looks).  `baker/radar_joint.py` refits all campaigns together with a
  height term shared by all of them, scores both calibrations on ice-free
  ground left out of the fit, and writes `radar_single/` and `radar_joint/`.
* **Prior and regularization**: `H = H_prior exp(theta_H)`, with `H_prior`
  from Farinotti et al. (2019); smoothness and size penalties on `theta_H`,
  smoothness on `q_bar` and on each data set's own `log N` terms, and size
  penalties on `q0` (spread `s_0`) and the 24 h terms (`baker/inverse.py`,
  `Regularization`).
* **All on one mesh**: every data set is put in UTM 10 N, on one rectangle
  covering every campaign's cells.

## Running

The data side runs where `gpri_tools` and the examples do; the model side
needs Firedrake, icepack and icepack2.

```sh
# 1. every campaign's strain rates, ITS_LIVE, and the grid (needs
#    GPRI_THICKNESS and GPRI_ITS_LIVE in site.env; see site.env.example)
python -m baker.prepare

# 2. each campaign's surface from the two antennas, then both calibrations
python -m baker.radar_dem --scene all
python -m baker.radar_joint

# 3. the inversion (Firedrake environment): fields.h5 and fields.npz;
#    radar campaigns only, each over its own radar surface
python -m baker.run --years --out radar_friction

# figures: the radar surfaces (base environment)
python -m baker.figures --radar --radar-source radar_single

# tests (the model tests skip without Firedrake)
python -m pytest baker/tests
```

Results go to `$GPRI_WORK_ROOT/baker/`.

**Building PyOP2 kernels with a conda compiler.**  If Firedrake cannot find
`mpicc`, put the environment's `bin` first on `PATH`.  If kernels then fail
to link, PyOP2 did not recognise the compiler and left out the shared-library
flags; set them:

```sh
export PYOP2_CFLAGS="-fPIC -O2 -march=native" PYOP2_LDFLAGS="-shared" OMP_NUM_THREADS=1
```
