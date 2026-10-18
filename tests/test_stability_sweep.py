from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from red_patterns.dct_dispersion import demo_pair
from red_patterns.stability_sweep import analyze_pair, ridge_pairs
from red_patterns.sweep_catalog import SweepCatalog, SweepEntry
from red_patterns.sweep_jobs import Gradient, PhiSweep, PhiType, TaylSweep, combine_sweeps


def catalog():
    runs = combine_sweeps(TaylSweep(
        N=[256], T=[30000], DT=[0.005], storeTime=[1],
        gradient=[Gradient.LINEAR_FULL], NU=[-8e-31, -8.1e-31], MU=[-4.5e-37],
        phi=PhiSweep(N=[256], psi_avg=[0.03], wing_z=[0], wing_r=[0],
                     phi_type=[PhiType.LINEAR_FULL_RIDGE, PhiType.PERTURBED_LINEAR_FULL_RIDGE],
                     mode_min=[1], mode_max=[100], epsilon=[1e-8]),
    ))
    return SweepCatalog(Path("/unused"), tuple(
        SweepEntry(run, Path("/unused") / run.run_id / "run.h5", True) for run in runs
    ))


def test_pairs_use_exact_parameters_not_run_order_or_default_float_tolerance():
    source = catalog()
    pairs = ridge_pairs(replace(source, entries=tuple(reversed(source.entries))))
    matches = {p["run_id"]: p["reference_id"] for p in pairs}
    assert matches == {"r000002": "r000001", "r000004": "r000003"}
    assert all(p["status"] == "ready" for p in pairs)


def test_missing_ambiguous_and_different_density_references_are_visible():
    source = catalog()
    missing = ridge_pairs(replace(source, entries=source.entries[1:]))
    assert missing[0]["status"] == "missing reference definition"
    duplicate = replace(source.entries[0], run=source.entries[0].run.model_copy(update={"run_id": "duplicate"}))
    ambiguous = ridge_pairs(replace(source, entries=source.entries + (duplicate,)))
    assert ambiguous[0]["status"] == "ambiguous reference definitions"
    absent = replace(source.entries[0], h5_exists=False)
    assert ridge_pairs(replace(source, entries=(absent,) + source.entries[1:]))[0]["status"] == "missing result file"
    run = source.entries[0].run.model_copy(deep=True)
    run.phi.params.psi_avg = 0.02
    other_density = replace(source.entries[0], run=run)
    assert ridge_pairs(replace(source, entries=(other_density,) + source.entries[1:]))[0]["status"] == "missing reference definition"


def test_known_growth_and_decay_and_unresolved_modes():
    perturbed, equilibrium = demo_pair()
    result = analyze_pair(perturbed, equilibrium, 6.01, 1, 9)
    assert result["summary"]["status"] == "resolved growth"
    assert result["summary"]["max_growth_s_inv"] == pytest.approx(0.16, abs=1e-7)
    assert result["summary"]["fit_end_s"] == pytest.approx(6)
    assert result["summary"]["fastest_mode"] == 5
    assert len(result["crossings"]) == 2
    assert analyze_pair(perturbed, equilibrium, 6, 8, 9)["summary"]["status"] == "selected modes decay"
    assert analyze_pair(perturbed, equilibrium, 6, 20, 23)["summary"]["status"] == "inconclusive"
    with pytest.raises(ValueError, match="five"):
        analyze_pair(perturbed, equilibrium, 0.1, 1, 9)
    with pytest.raises(ValueError, match="mode range"):
        analyze_pair(perturbed, equilibrium, 6, 1, 100)


def test_reference_drift_blocks_classification_and_theory_uses_saved_values():
    perturbed, equilibrium = demo_pair()
    for run in (perturbed, equilibrium):
        run["model"] = dict(modelType="TAYL", gradientType="LINEAR_FULL", alpha=2e-5, beta=7.4e23)
        run["variant"] = dict(NU=-8e-31, MU=-4.5e-37)
    equilibrium["psi"][-1] += 1e-4
    result = analyze_pair(perturbed, equilibrium, 6, 1, 9)
    s = result["summary"]
    assert s["status"] == "check reference / metadata"
    assert result["notes"]
    assert s["g"] == pytest.approx(30 / 0.064)
    assert s["psi_avg"] == pytest.approx(0.02)
    assert s["nu_critical"] == pytest.approx(-np.sqrt(-4 * s["alpha"] * s["g"] * s["MU"] / (s["beta"] * s["psi_avg"])), rel=1e-12, abs=1e-45)
    k = result["table"]["wavenumber_rad_m"].to_numpy()
    np.testing.assert_allclose(result["table"]["theory_rate_s^-1"],
                               -s["alpha"] * s["g"] - s["beta"] * s["psi_avg"] * s["NU"] * k**2
                               + s["beta"] * s["psi_avg"] * s["MU"] * k**4)
