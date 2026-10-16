"""The copies of the figures docs/flow.md shows, under their names there."""
import pytest

from baker.figures import docs_figures

CAMPAIGNS = ("20170803_full", "20190719")


def _draw(path, root):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(path.relative_to(root)))


def _drawn(root):
    folder = root / "baker"
    run = folder / "radar_friction"
    for name in CAMPAIGNS:
        _draw(folder / "radar_single" / f"radar_dh_{name}.npz", root)
        for kind in ("dh", "sd"):
            _draw(folder / "fig" / "radar_single" / f"radar_{kind}_{name}.png", root)
        _draw(run / "fig" / f"dlogn_{name}.png", root)
    for source in ("radar_single", "radar_joint"):
        _draw(folder / "fig" / source / "radar_dh_elevation.png", root)
    for kind in ("amp", "peak"):
        _draw(run / "fig" / f"logn_{kind}_20190719.png", root)
    for k in ("thickness", "bed", "n_stationary", "misfit", "speed_model_20190719"):
        _draw(run / "fig" / f"{k}.png", root)
    return folder, run


def test_each_figure_lands_under_its_name_on_the_page(tmp_path):
    folder, run = _drawn(tmp_path)
    dest = tmp_path / "docs" / "figures"
    written = docs_figures(folder, run, dest)
    assert sorted(p.name for p in dest.iterdir()) == sorted(p.name for p in written) == sorted([
        "42_radar_dh_elevation.png", "42_radar_dh_elevation_joint.png",
        "43_radar_dh_20170803_full.png", "43_radar_dh_20190719.png",
        "43_radar_sd_20170803_full.png",
        "44_flow_thickness.png", "44_flow_bed.png", "45_flow_n_stationary.png",
        "46_flow_dlogn_20170803_full.png", "46_flow_dlogn_20190719.png",
        "47_flow_logn_amp_20190719.png", "47_flow_logn_peak_20190719.png",
        "48_flow_misfit.png"])
    assert (dest / "42_radar_dh_elevation_joint.png").read_text() == \
        "baker/fig/radar_joint/radar_dh_elevation.png"
    assert (dest / "46_flow_dlogn_20190719.png").read_text() == \
        "baker/radar_friction/fig/dlogn_20190719.png"


def test_a_figure_not_drawn_yet_is_named_and_nothing_is_copied(tmp_path):
    folder, run = _drawn(tmp_path)
    (run / "fig" / "bed.png").unlink()
    dest = tmp_path / "docs" / "figures"
    with pytest.raises(SystemExit, match="bed.png"):
        docs_figures(folder, run, dest)
    assert not dest.exists()
