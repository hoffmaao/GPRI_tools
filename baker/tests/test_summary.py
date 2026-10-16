"""The numbers docs/flow.md quotes: spread between campaigns, values at cells, the run log."""
import numpy as np

from baker.summary import at_cells, distinct_cells, read_log, spread


def test_spread_is_the_sd_across_campaigns_on_cells_seen_by_all():
    nan = np.nan
    dh = np.array([[[0.0, 0.0, 0.0, 5.0]],
                   [[2.0, 1.0, nan, 5.0]],
                   [[4.0, 2.0, 9.0, 5.0]]])          # (campaigns, rows, cols)
    z = np.array([[1600.0, 2100.0, 1600.0, 1600.0]])
    ice = np.array([[True, True, True, False]])
    n, rows = spread(dh, z, ice, {"all": [0, 1, 2], "first two": [0, 1]},
                     bands=((1500.0, 2000.0), (2000.0, 2300.0), (2300.0, 2600.0)))
    assert n == 2                         # the gap and the cell off the ice are left out
    (lo, hi, cells, low), (_, _, cells2, mid), (_, _, cells3, high) = rows
    assert (lo, hi, cells, cells2, cells3) == (1500.0, 2000.0, 1, 1, 0)
    assert np.isclose(low["all"], np.std([0.0, 2.0, 4.0]))
    assert np.isclose(low["first two"], 1.0)
    assert np.isclose(mid["all"], np.std([0.0, 1.0, 2.0]))
    assert np.isnan(high["all"])


def test_at_cells_takes_the_nearest_grid_cell():
    x = np.array([0.0, 50.0, 100.0])
    y = np.array([1000.0, 1050.0])
    field = np.arange(6.0).reshape(2, 3)      # rows along y
    xy = np.array([[49.0, 1001.0], [99.0, 1049.0], [-10.0, 1100.0]])
    assert at_cells(field, x, y, xy).tolist() == [1.0, 5.0, 3.0]


def test_distinct_cells_counts_a_grid_cell_seen_by_several_campaigns_once():
    x = np.array([0.0, 50.0, 100.0])
    y = np.array([1000.0, 1050.0])
    first = np.array([[49.0, 1001.0], [99.0, 1049.0]])
    second = np.array([[52.0, 998.0], [1.0, 1000.0]])     # a few metres from the first's first cell
    j, i = distinct_cells(x, y, [first, second])
    assert sorted(zip(j.tolist(), i.tolist())) == [(0, 0), (0, 1), (1, 2)]


def test_read_log_takes_the_objective_and_each_misfit(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("7 data sets, 30410 observations\n"
                   "iteration   0  J 38107.8  misfit/obs: a_full 0.535  b 16.6\n"
                   "iteration   1  J 29502.8  misfit/obs: a_full 0.445  b 12\n"
                   "-> fields.h5\n")
    its, J, misfit = read_log(log)
    assert its.tolist() == [0, 1]
    assert J.tolist() == [38107.8, 29502.8]
    assert {k: v.tolist() for k, v in misfit.items()} == {"a_full": [0.535, 0.445], "b": [16.6, 12.0]}
