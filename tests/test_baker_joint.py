"""Where examples/baker_joint.py writes its results and draws its figures."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
sys.path.append(str(Path(__file__).resolve().parents[1] / "examples"))
import baker_joint  # noqa: E402


def _folders(*argv):
    return baker_joint.folders(baker_joint.parser().parse_args(list(argv)))


def test_the_default_settings_draw_into_the_docs():
    results, figures = _folders()
    assert results.name == "z2300_global"
    assert figures == Path("docs/figures")


@pytest.mark.parametrize("argv", [("--cand-min-z", "0"), ("--shared-noise", "cell")])
def test_a_comparison_setting_draws_beside_its_own_results(argv):
    results, figures = _folders(*argv)
    assert results != _folders()[0]
    assert figures == results / "figures"


def test_an_explicit_outdir_is_kept(tmp_path):
    assert _folders("--cand-min-z", "0", "--outdir", str(tmp_path))[1] == tmp_path
