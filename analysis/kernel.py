# /// script
# dependencies = [
#     "h5py==3.16.0",
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
#     "scipy==1.17.1",
#     "wigglystuff==0.3.3",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import h5py
    import marimo as mo
    import numpy as np
    from pathlib import Path
    from scipy.integrate import cumulative_trapezoid
    import matplotlib.pyplot as plt
    from dataclasses import dataclass
    from typing import Callable, ClassVar, Dict
    from wigglystuff import CopyToClipboard

    return (
        Callable,
        ClassVar,
        CopyToClipboard,
        Dict,
        Path,
        cumulative_trapezoid,
        dataclass,
        h5py,
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Kernel

    This notebook constructs the effective one-dimensional interaction kernel
    $K$ that appears in the reduced DDFT model.

    $$
        \partial_t \varphi(\rho,z,t) + \partial_z J_z(\rho,z,t) = 0
    $$

    with flux

    $$
        J_z(\rho,z,t) = \Gamma\,\varphi(\rho,z,t) \left( \frac{1}{V} \int_0^L \psi(z',t)\,K(z-z')\,d z' - \partial_z u_{\mathrm{ext}}(\rho,z,t) \right).
    $$

    The total volume fraction is

    $$
        \psi(z,t) := \int_M \varphi(\rho,z,t)\, d\rho.
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The reduced model admits two common closures:

    1. Potential closure

    $$
        K(x) = 2\pi\,x\,g(|x|)\,u(|x|)
    $$

    2. Force closure
        $$
            K(x) = 2\pi\,x\int_{|x|}^{\infty} g(R)\,f(R)\,d R,
        $$
        where for conservative forces $f(R) = -u'(R)$.

    In the mean-field case $g=1$, both closures coincide. Otherwise they can
    produce visibly different kernels.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pair Potential

    For now we use a Lennard-Jones pair potential

    $$
        u(r) = 4U \left(\frac{\sigma^{12}}{r^{12}} - \frac{\sigma^6}{r^6}\right),
    $$

    with $\sigma = 5.6\,\mu\mathrm{m}$ and $U = 111.15 \times 10^{-18}\,\mathrm{J}$.

    The pair distribution function $g$ models correlations between cells. The
    closure then determines how $u$, $g$, and the pair force enter the kernel.
    """)
    return


@app.cell
def _(np):
    SIGMA = 5.6e-6  # 5.6 micrometers converted to meters
    U = 111.15e-18  # 111.15 * 10^-18 Joules
    V = 90e-18  # 90 fL in m^3

    def lennard_jones_potential(r: np.ndarray):
        return 4 * U * ((SIGMA / r) ** 12 - (SIGMA / r) ** 6)

    return SIGMA, U, lennard_jones_potential


@app.cell
def _(SIGMA, lennard_jones_potential, mo, np, plt):
    _r = np.linspace(0.95 * SIGMA, 3 * SIGMA, 500)
    u = lennard_jones_potential(_r)

    # 4. Convert units for cleaner axis labels
    # Convert r to micrometers (um) and u to 10^-18 Joules (aJ)
    r_um = _r * 1e6
    u_aJ = u * 1e18

    # 5. Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(r_um, u_aJ, color="blue", linewidth=2, label="Lennard-Jones potential")

    # Add a horizontal line at y=0 and vertical line at r=sigma
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(SIGMA * 1e6, color="red", linestyle="--", label=r"$\sigma = 5.6 \mu m$")

    # 6. Add labels, title, and limits
    plt.xlabel(r"Distance $r$ ($\mu$m)", fontsize=12)
    plt.ylabel(r"Potential $u(r)$ ($10^{-18}$ J)", fontsize=12)
    plt.title("Lennard-Jones potential", fontsize=14)

    # Restrict the y-axis so the steep curve doesn't squash the potential well
    plt.ylim(-150, 100)

    # 7. Add grid and legend
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.axhline(0, color="white", linewidth=1, linestyle="-")

    _lj_plot = mo.ui.matplotlib(plt.gca())
    _lj_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pair Distribution Function

    The pair distribution function $g(r)$ captures how strongly nearby RBCs are
    correlated. Use the tabs to choose one of the available approximations.
    """)
    return


@app.cell
def _(Callable, ClassVar, Dict, U, dataclass, lennard_jones_potential, mo, np):
    @dataclass
    class PairDistributionObject:
        key: str
        markdown: mo.Html
        func: Callable[[np.ndarray], np.ndarray]
        registry: ClassVar[Dict[str, "PairDistributionObject"]] = {}

        def __post_init__(self):
            self.registry[self.key] = self

    # Pair Distribution Objects

    def _with_guard(
        fn: Callable[[np.ndarray], np.ndarray],
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Wrap a pair distribution with the legacy divergence guard.

        The original kernel generator sets g(r)=0 for r<1e-8 to avoid
        numerical issues near the LJ divergence.
        """

        def guarded(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            out = np.asarray(fn(x), dtype=np.float64)
            out = out.copy()
            out[x < 1e-8] = 0.0
            return out

        return guarded

    _ = PairDistributionObject(
        key="Mean Field",
        markdown=mo.md(
            r"""
    The mean-field approximation assumes no positional correlations.

    $$
        g(x) = 1
    $$
    """
        ),
        func=_with_guard(lambda x: np.ones_like(x)),
    )

    G0 = 4.0e7
    SIGMA_C = 0.5e-6
    EQ_DIST = 6.585467201064237091254725819933213415424688719213008880615234375e-06

    _ = PairDistributionObject(
        key="Nearest Neighbor",
        markdown=mo.md(
            r"""
    This approximation concentrates the weight around one preferred RBC spacing.

    $$
        g(x) = g_0 \exp\left(-\frac{(r-d)^2}{2\sigma_C^2}\right)
    $$
    """
        ),
        func=_with_guard(
            lambda x: G0 * np.exp(-((x - EQ_DIST) ** 2) / (2 * (SIGMA_C**2)))
        ),
    )

    LAMBDA = 1

    _ = PairDistributionObject(
        key="Exponential",
        markdown=mo.md(
            r"""
    This ansatz reuses the pair potential itself to suppress strongly repulsive configurations.

    $$
        g(x) = \exp \left( -\lambda \frac{u(x)}{U} \right)
    $$
    """
        ),
        func=_with_guard(lambda x: np.exp(-LAMBDA * (lennard_jones_potential(x) / U))),
    )

    # TODO: Makr Custom PDF?
    return (PairDistributionObject,)


@app.cell
def _(PairDistributionObject, SIGMA, np):
    """Build the discrete convolution stencil consumed by CUDA.

    This is the single source of truth for both plotting and export.
    For (Force closure, Nearest Neighbor) it matches the legacy CUDA
    implementation in `src/simulations.cu::genConvKernel` and therefore
    `createKernel.py`.
    """

    # Legacy sampling resolution baked into the original kernel generator.
    _LEGACY_SUB_RES = 10000.0

    def _f_lj(r: np.ndarray, sigma: float, u_scale: float) -> np.ndarray:
        # Matches CUDA: 4U*(12*sigma^12/r^13 - 6*sigma^6/r^7)
        return (
            4.0 * u_scale * (12.0 * (sigma**12) / (r**13) - 6.0 * (sigma**6) / (r**7))
        )

    def _u_lj(r: np.ndarray, sigma: float, u_scale: float) -> np.ndarray:
        # Matches notebook potential definition.
        return 4.0 * u_scale * ((sigma / r) ** 12 - (sigma / r) ** 6)

    def build_kernel_stencil(
        *,
        closure: str,
        pair_distribution_key: str,
        kernel_n: int,
        dz: float,
        sub_div: int,
        u_scale: float,
        sigma: float = SIGMA,
        sub_res: float = _LEGACY_SUB_RES,
    ):
        if kernel_n < 3 or (kernel_n % 2) == 0:
            raise ValueError("kernelN must be an odd integer >= 3")
        if dz <= 0.0:
            raise ValueError("DZ must be positive")
        if sub_div <= 0:
            raise ValueError("subDiv must be a positive integer")

        dz_up = dz / float(sub_div)
        center = (kernel_n - 1) // 2
        x = (np.arange(kernel_n, dtype=np.float64) - center) * dz_up

        kernel_l = (kernel_n - 1) * dz_up
        kernel_dz = dz_up

        # Fine radial grid matches legacy genConvKernel scheme.
        fine_res = int(sub_res * ((kernel_n + 1) / 2))
        fine_dr = kernel_dz / sub_res
        r = np.arange(fine_res, dtype=np.float64) * fine_dr

        # Pair distribution (plot and export must agree).
        g_fn = PairDistributionObject.registry[pair_distribution_key].func
        g_r = np.asarray(g_fn(r), dtype=np.float64)

        kernel_values = np.zeros(kernel_n, dtype=np.float64)
        kernel_values[center] = 0.0

        if closure == "Force closure":
            # Exact legacy force closure discretization.
            # kernelFine[j] = sum_{k<j} fine_dr * fLJ(r_k)*g(r_k)
            # then flipped to represent the tail integral in the same discrete sense.
            kernel_fine = np.zeros(fine_res, dtype=np.float64)
            force_sum = 0.0
            # Start from 1 to match legacy kernelFine[0]=0 and loop i=1..fineRes-1
            for i in range(1, fine_res):
                fine_r = r[i]
                kernel_fine[i] = force_sum
                # Analytic LJ force, matches CUDA.
                force_sum += (
                    fine_dr * float(_f_lj(fine_r, sigma, u_scale)) * float(g_r[i])
                )

            kernel_fine = kernel_fine[-1] - kernel_fine

            for i in range(center + 1, kernel_n):
                kernel_z = float(i) * kernel_dz - kernel_l / 2.0
                # Matches CUDA: (i + 1 - (center + 1)) * subRes == (i-center)*subRes
                sample_idx = int((i - center) * sub_res)
                kernel_values[i] = kernel_z * kernel_fine[sample_idx]
                kernel_values[kernel_n - 1 - i] = -kernel_values[i]

        elif closure == "Potential closure":
            # Sample g(|x|) * u(|x|) on the same stencil grid.
            r_abs = np.abs(x)
            # Avoid r=0 singularity by clipping to the smallest representable r.
            r_abs = np.maximum(r_abs, fine_dr)

            g_x = np.asarray(g_fn(r_abs), dtype=np.float64)

            u_x = np.asarray(_u_lj(r_abs, sigma, u_scale), dtype=np.float64)
            # Notebook convention includes 2πx; keep that convention for potential closure.
            kernel_values = 2.0 * np.pi * x * (g_x * u_x)
            kernel_values[center] = 0.0
        else:
            raise ValueError(f"Unknown closure: {closure}")

        return x, kernel_values, kernel_dz

    return (build_kernel_stencil,)


@app.cell
def _(PairDistributionObject, mo):
    pair_distribution_tabs = mo.ui.tabs(
        {key: node.markdown for key, node in PairDistributionObject.registry.items()},
        value="Nearest Neighbor",
    )

    pair_distribution_tabs
    return (pair_distribution_tabs,)


@app.cell
def _(PairDistributionObject, mo, np, pair_distribution_tabs, plt):
    active_pair_distribution = PairDistributionObject.registry[
        pair_distribution_tabs.value
    ]

    r_plot = np.linspace(1e-9, 5e-5, 400)

    plt.figure(figsize=(8, 6))
    plt.plot(
        r_plot * 1e6,
        active_pair_distribution.func(r_plot),
        color="blue",
        linewidth=2,
    )

    plt.xlabel(r"Distance $r$ ($\mu$m)", fontsize=12)
    plt.ylabel(r"Pair distribution $g(r)$", fontsize=12)
    plt.title("Pair distribution function", fontsize=14)

    plt.grid(True, linestyle=":", alpha=0.7)

    _pdf_plot = mo.ui.matplotlib(plt.gca())
    _pdf_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Closure Options

    Depending on where you introduce the Pair-Distribution-Function $g$, you get different closures and therefor different Kernels.

    1. If you choose $g$ to be the closure of the two-body-density, and assume that it is independent of the one-body-density you get
        $$
        u_{eff} = g \cdot u
        $$
        which is the *Potential Closure of the Kernel*.
    2. If you choose $g$ to describe, the effective interaction force $f_{eff}$ in terms of the "normal" interaction force $f$

        $$
        f_{eff} = g \cdot f
        $$

        you get the *Force Closure of the Kernel*.
    """)
    return


@app.cell
def _(mo):
    closure_tabs = mo.ui.tabs(
        {
            "Potential closure": mo.md(
                r"""
                $$
                K(x) = 2\pi\,x\,g(|x|)\,u(|x|)
                $$
                """
            ),
            "Force closure": mo.md(
                r"""
                $$
                K(x) = 2\pi\,x\int_{|x|}^{\infty} g(R)\,f(R)\, d R,
                \qquad f(R) = -u'(R)
                $$
                """
            ),
        },
        value="Force closure",
    )
    closure_tabs
    return (closure_tabs,)


@app.cell
def _(mo):
    # Defaults match the legacy CUDA generator.
    plot_u = mo.ui.number(
        start=0.0,
        step=1.0,
        value=100.0,
        label="U (1e-18 J)",
    )
    plot_kernel_n = mo.ui.number(start=3, stop=10001, step=2, value=31, label="kernelN")
    plot_dz = mo.ui.number(
        start=1e-12,
        step=1e-7,
        # Legacy default: matches `createKernel.py` (see IZ derivation there).
        value=256.0 * 1.0455122765372783e-6,
        label="DZ",
    )
    plot_sub_div = mo.ui.number(start=1, step=1, value=256, label="subDiv")

    controls = mo.md("""
    Kernel parameters (used for both plotting and export sampling):

    {plot_u}

    {plot_kernel_n}

    {plot_dz}

    {plot_sub_div}
    """).batch(
        plot_u=plot_u,
        plot_kernel_n=plot_kernel_n,
        plot_dz=plot_dz,
        plot_sub_div=plot_sub_div,
    )

    controls
    # Downstream cells (moments) should use the exported stencil.
    return (controls,)


@app.cell
def _(
    PairDistributionObject,
    build_kernel_stencil,
    closure_tabs,
    controls,
    cumulative_trapezoid,
    mo,
    np,
    pair_distribution_tabs,
    plt,
):
    closure_name = str(closure_tabs.value)
    pair_key = str(pair_distribution_tabs.value)

    kernel_n = int(controls.value["plot_kernel_n"])
    dz = float(controls.value["plot_dz"])
    sub_div = int(controls.value["plot_sub_div"])
    # UI uses 1e-18 J units for convenience.
    u_scale = float(controls.value["plot_u"]) * 1e-18

    x_stencil, k_stencil, spacing = build_kernel_stencil(
        closure=closure_name,
        pair_distribution_key=pair_key,
        kernel_n=kernel_n,
        dz=dz,
        sub_div=sub_div,
        u_scale=u_scale,
    )

    # Smooth curve for visualization.
    x_max = max(abs(float(x_stencil[0])), abs(float(x_stencil[-1])))
    x_plot = np.linspace(-x_max, x_max, 2001)
    r = np.linspace(spacing / 10000.0, x_max, 20000)

    # This is used only for plotting; export is always from the stencil builder.
    if pair_key == "Nearest Neighbor":
        g0 = 4.0e7
        sigma_c = 0.5e-6
        eq_dist = 6.585467201064237091254725819933213415424688719213008880615234375e-6
        g_r = g0 * np.exp(-((r - eq_dist) ** 2) / (2.0 * sigma_c**2))
        g_r = g_r.copy()
        g_r[r < 1e-8] = 0.0
    else:
        g_r = np.asarray(
            PairDistributionObject.registry[pair_key].func(r), dtype=np.float64
        )

    sigma = 5.6e-6
    u_r = 4.0 * u_scale * ((sigma / r) ** 12 - (sigma / r) ** 6)
    f_r = 4.0 * u_scale * (12.0 * sigma**12 / (r**13) - 6.0 * sigma**6 / (r**7))

    if closure_name == "Potential closure":
        radial = g_r * u_r
        k_plot = 2.0 * np.pi * x_plot * np.interp(np.abs(x_plot), r, radial)
    else:
        integrand = g_r * f_r
        tail = -cumulative_trapezoid(integrand[::-1], r[::-1], initial=0.0)[::-1]
        k_plot = 2.0 * np.pi * x_plot * np.interp(np.abs(x_plot), r, tail)

    _fig, _ax = plt.subplots(figsize=(8, 6))
    _ax.plot(x_plot * 1e6, k_plot, color="blue", linewidth=2, label="continuous")
    _ax.scatter(
        x_stencil * 1e6,
        k_stencil,
        s=18,
        color="black",
        label="exported stencil",
        zorder=3,
    )
    _ax.axhline(0, color="black", linewidth=1)
    _ax.set_xlabel(r"Offset $x$ ($\mu$m)", fontsize=12)
    _ax.set_ylabel(r"Kernel $K(x)$", fontsize=12)
    _ax.set_title(f"{closure_name} ({pair_key})", fontsize=14)
    _ax.grid(True, linestyle=":", alpha=0.7)
    _ax.legend()

    mo.ui.matplotlib(_ax)
    return k_stencil, x_stencil


@app.cell
def _(k_stencil, x_stencil):
    x = x_stencil
    kernel_values = k_stencil
    return (kernel_values,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CUDA Export

    This export form writes the **discrete convolution stencil** expected by the
    CUDA code. The exported stencil is built from the currently selected closure
    and pair distribution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How The Simulation Samples The Kernel

    The CUDA simulation does not consume a continuous kernel $K(x)$. Instead, it
    uses a discrete odd-length stencil

    $$
    \,\{K_i\}_{i=-(N_K-1)/2}^{(N_K-1)/2}
    $$

    with `kernelN = N_K` samples and spacing

    $$
    \Delta z_K = \frac{DZ}{\text{subDiv}}.
    $$

    The sampled offsets are therefore

    $$
    x_i = i\,\Delta z_K,
    \qquad
    i = -\frac{N_K-1}{2}, \ldots, \frac{N_K-1}{2}.
    $$

    so the total sampled support is

    $$
    L_K = (N_K - 1)\,\Delta z_K.
    $$

    During the convolution step, CUDA multiplies these discrete kernel entries by
    the interpolated density values and then applies one additional factor of

    $$
    \Delta z_K = \frac{DZ}{\text{subDiv}}
    $$

    outside the summation. That means the exported `/kernel/values` dataset must
    already be sampled on this exact grid to reproduce the legacy simulation
    behavior.
    """)
    return


@app.cell
def _(Path, mo):
    export_dir = mo.ui.file_browser(
        initial_path=Path.cwd(),
        ignore_empty_dirs=False,
        multiple=False,
        selection_mode="directory",
        label="Export directory",
    )
    export_name = mo.ui.text(value="kernel.h5", label="File name")
    export_form = (
        mo.md(
            """
    Export a CUDA-compatible convolution kernel.

    {export_dir}

    {export_name}
    """
        )
        .batch(
            export_dir=export_dir,
            export_name=export_name,
        )
        .form(
            submit_button_label="Export CUDA kernel",
            clear_on_submit=False,
            show_clear_button=True,
        )
    )
    export_form
    return (export_form,)


@app.cell
def _(
    CopyToClipboard,
    Path,
    build_kernel_stencil,
    closure_tabs,
    controls,
    export_form,
    h5py,
    mo,
    np,
    pair_distribution_tabs,
):
    if export_form.value is None:
        _result = mo.md(
            "Submit the form to export a CUDA-compatible convolution kernel."
        )
    else:
        _export_dir_entries = export_form.value.get("export_dir") or []
        _file_name = str(export_form.value.get("export_name", "")).strip()
        # Always export with the same live parameters used for plotting.
        _kernel_n = int(controls.value["plot_kernel_n"])
        _dz = float(controls.value["plot_dz"])
        _sub_div = int(controls.value["plot_sub_div"])
        # UI uses 1e-18 J units for convenience.
        _u_scale = float(controls.value["plot_u"]) * 1e-18

        if not _export_dir_entries:
            _result = mo.md("Please select a directory before exporting.")
        elif not _file_name:
            _result = mo.md("Please enter a file name before exporting.")
        else:
            _selected_dir = Path(_export_dir_entries[0].path)
            _path = _selected_dir / _file_name

            _closure = str(closure_tabs.value)
            _pair_key = str(pair_distribution_tabs.value)

            _x, _kernel_values, _kernel_dz = build_kernel_stencil(
                closure=_closure,
                pair_distribution_key=_pair_key,
                kernel_n=_kernel_n,
                dz=_dz,
                sub_div=_sub_div,
                u_scale=_u_scale,
            )

            try:
                _path.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(_path, "w") as _f:
                    _kernel_group = _f.create_group("kernel")
                    _kernel_group.create_dataset(
                        "values", data=np.asarray(_kernel_values, dtype=np.float64)
                    )
                    _kernel_group.create_dataset(
                        "x", data=np.asarray(_x, dtype=np.float64)
                    )
                    _kernel_group.attrs["kernelN"] = int(_kernel_values.shape[0])
                    _kernel_group.attrs["spacing"] = float(_kernel_dz)
                    _kernel_group.attrs["DZ"] = float(_dz)
                    _kernel_group.attrs["subDiv"] = int(_sub_div)
                    _kernel_group.attrs["closure"] = _closure
                    _kernel_group.attrs["pair_distribution"] = _pair_key
                    _kernel_group.attrs["U"] = float(_u_scale)
                    _kernel_group.attrs["cuda_compatible"] = 1
                    _kernel_group.attrs["generated_by"] = "analysis/kernel.py"
            except Exception as _exc:
                _result = mo.md(rf"Export failed: `{_exc}`")
            else:
                _clipboard = mo.ui.anywidget(CopyToClipboard(text_to_copy=str(_path)))
                _result = mo.vstack(
                    [
                        mo.md(
                            rf"Exported CUDA-compatible convolution kernel to `{_path}`."
                        ),
                        mo.md(
                            rf"Stored `{_kernel_n}` samples with spacing `{_kernel_dz:.6e}` m on `/kernel/values`."
                        ),
                        mo.md(
                            rf"Used live plotting parameters: `kernelN={_kernel_n}`, `DZ={_dz:.6e}`, `subDiv={_sub_div}`, `U={_u_scale:.6e}` (J)."
                        ),
                        mo.hstack(
                            [mo.md("Copy exported path:"), _clipboard], justify="start"
                        ),
                    ]
                )

    _result
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Taylor Approximation

    We approximate

    $$
        I(z,t) = \frac{1}{V} \int_{0}^{L} d z' \, \psi(z', t) K(z - z')
    $$

    by expanding $\psi(z',t)$ around $z$ and introducing $\eta = z-z'$. This gives

    $$
        I(z,t)
        = -\frac{1}{V}\sum_{n=0}^{\infty}
        \frac{\partial_z^{2n+1}\psi(z,t)}{(2n+1)!}
        \int_{-\infty}^{+\infty} d\eta \, K(\eta)\eta^{2n+1},
    $$

    where the odd-kernel symmetry removes all even orders on a symmetric domain, giving us

    $$
        I(z,t) \approx -\frac{1}{V}\partial_z \psi(z, t) \int_{-\infty}^{+\infty} d{\eta}\, K(\eta)\eta - \frac{1}{V}\frac{1}{3!}\partial_z^3 \psi(z, t) \int_{-\infty}^{+\infty} d{\eta}\, K(\eta)\eta^3 + \ldots
    $$

    with moments

    $$
    \begin{aligned}
        \nu &=- \frac{1}{V} \int_{-\infty}^{+\infty} d\eta \, K(\eta)\eta, \\
        \mu &=- \frac{1}{V} \frac{1}{3!}\int_{-\infty}^{+\infty} d\eta \, K(\eta)\eta^3.
    \end{aligned}
    $$

    so that

    $$
        I(z,t) \approx \nu \, \partial_z \psi(z,t) + \mu \, \partial_z^3 \psi(z,t).
    $$
    """)
    return


@app.cell
def _(controls, kernel_values, mo, np):
    #   NU = int r K(r) dr        MU = (1/6) int r^3 K(r) dr
    # with a discrete Riemann sum on a grid of spacing kernelDZ = DZ/256.
    _dz = float(controls.value["plot_dz"])
    sub_div_tayl = 256
    kernel_dz = _dz / float(sub_div_tayl)

    _kernel_n = int(kernel_values.shape[0])
    centre = (_kernel_n - 1) // 2
    rk = (np.arange(_kernel_n, dtype=np.float64) - float(centre)) * kernel_dz

    nu = kernel_dz * float(np.sum(rk * kernel_values))
    mu = (kernel_dz * float(np.sum((rk**3) * kernel_values))) / 6.0

    def latex_scientific(value: float) -> str:
        if value == 0:
            return "0"
        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / (10**exponent)
        return rf"{mantissa:.6f} \cdot 10^{{{exponent}}}"

    mo.md(
        rf"""
    ### Numerical Coefficients

    $$
    \begin{{aligned}}
    \mathrm{{NU}} &= {latex_scientific(nu)}, \\
    \mathrm{{MU}} &= {latex_scientific(mu)}.
    \end{{aligned}}
    $$
    """
    )
    return


if __name__ == "__main__":
    app.run()
