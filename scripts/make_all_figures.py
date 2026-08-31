"""Master figure-and-table generator for the journal paper.

This script consolidates make_figures_and_tables.py and make_trajectory_figures.py
into a single entry point. Generates all paper figures (8 total) and 4 LaTeX tables.

USAGE:
    python make_all_figures.py [--skip-traj]

OPTIONS:
    --skip-traj   Skip trajectory simulation (faster, only stats figures).
                  Use this if you've already run trajectory simulations and
                  just want to refresh the statistical figures.

OUTPUT:
    paper_figures/
        # Statistical figures (from results JSONs):
        figure03_error_cdfs.{pdf,png}        - CDFs
        figure04_per_tumble_breakdown.{pdf,png}        - per-tumble box plots
        figure02_conformal_calibration.{pdf,png}       - q_hat + coverage validation
        figure05_capture_scatter.{pdf,png}             - capture-ready scatter
        figure06_rejection_stratified.{pdf,png}       - rejection vs capture rate

        # Trajectory figures (from live simulation):
        figure07_trajectory_3d.{pdf,png}               - 3D paths
        figure08_position_tracking_error.{pdf,png}     - per-axis tracking error
        figure09_tracking_errors_envelope.{pdf,png}            - errors success vs failure
        figure10_uncertainty_management.{pdf,png}     - q_hat + gain modulation

        # LaTeX-ready tables:
        table2_overall_results.tex
        table1_calibration_constants.tex
        table3_per_tumble.tex
        table4_capture_thresholds.tex

INPUTS REQUIRED:
    - path2_winning_results.json   (Phase 1 baseline)
    - path2_conformal_results.json (Conformal-aware run)
    - conformal_calibration.json   (calibrated q_hat)
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=UserWarning, module="cap_control")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Hamiltonian.*")

OUT_DIR = "paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Style: clean academic look
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 1.3,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.6",
})


def legend_below(ax, ncol, yoff=-0.30, **kw):
    """Place this axes' legend centered below the axes (paper style)."""
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, yoff),
                     ncol=ncol, frameon=True, **kw)


def fig_legend_below(fig, handles, labels, ncol, y=0.0, **kw):
    """One shared legend centered below the whole figure."""
    return fig.legend(handles, labels, loc="upper center",
                      bbox_to_anchor=(0.5, y), ncol=ncol, frameon=True, **kw)

COLOR_BASELINE = "#5B6CFF"
COLOR_CONFORMAL = "#E63946"
COLOR_EE = "#1f77b4"
COLOR_GRASP = "#2ca02c"
COLOR_UNCERTAINTY = "#ff7f0e"


# ============================================================================
# Helpers
# ============================================================================
def load_results(path):
    with open(path) as f:
        data = json.load(f)
    out = {"omega": [], "pos_tf": [], "ori_tf": [], "pos_last1s": [],
           "ori_last1s": [], "n_rejected": [], "max_q_hat": [], "success": []}
    for r in data:
        if not r["success"]:
            continue
        out["omega"].append(r["omega"])
        out["pos_tf"].append(r["pos_tf"])
        out["ori_tf"].append(r["ori_tf"])
        out["pos_last1s"].append(r["pos_last1s"])
        out["ori_last1s"].append(r["ori_last1s"])
        out["n_rejected"].append(r.get("n_rejected", 0))
        out["max_q_hat"].append(r.get("max_q_hat", 0.0))
        out["success"].append(True)
    for k in out:
        out[k] = np.array(out[k])
    out["n_total"] = len(data)  # includes diverged runs (counted as failures in rates)
    return out


def load_test_coverage(path="calibration_comparison.json"):
    """Empirical coverage on the FULL held-out test set (45 trajectories),
    produced by diagnose_paper_consistency.py (Section E). Copy the file from
    the diagnosis_out folder into this directory before running."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            "calibration_comparison.json not found. Copy it from the "
            "diagnosis_out folder (created by diagnose_paper_consistency.py) "
            "into the directory you run make_all_figures.py from.")
    with open(path) as f:
        d = json.load(f)["repeated_subsample"]["per_lookahead"]
    return {la: {"coverage_ori_pct": v["cov_ori_marginal"],
                  "coverage_pos_pct": v["cov_pos_marginal"],
                  "ci_ori": v["ci_ori"], "ci_pos": v["ci_pos"]}
            for la, v in d.items()}


def quat_angle_deg(q1, q2):
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    q2 = q2 / max(np.linalg.norm(q2), 1e-12)
    cw, cx, cy, cz = q1[0], -q1[1], -q1[2], -q1[3]
    qw, qx, qy, qz = q2
    rw = qw * cw - qx * cx - qy * cy - qz * cz
    return 2 * np.rad2deg(np.arccos(min(1.0, abs(rw))))


def kp_ori_from_uncertainty(qb, q_low=20.0, q_high=40.0,
                              kp_max=5.0, kp_min=1.0):
    frac = float(np.clip((qb - q_low) / (q_high - q_low), 0.0, 1.0))
    sm = frac * frac * (3.0 - 2.0 * frac)
    return float(kp_max - (kp_max - kp_min) * sm)


# ============================================================================
# Statistical figures (always run)
# ============================================================================
def generate_statistical_figures(phase1, conf, calib):
    """Figures 1, 2, 3, 4, 6 + tables."""

    # -----------------------------------------------------------------------
    # FIGURE 2: Conformal calibration with CIs and acceptable band
    # -----------------------------------------------------------------------
    print("\nGenerating Figure 2: Conformal calibration...")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2))
    las = [0.5, 1.0, 1.5, 2.0]
    qo = [calib["q_hat_orientation_deg"][str(la)] for la in las]
    qp = [calib["q_hat_position_cm"][str(la)] for la in las]
    covd = load_test_coverage()
    test_cov_ori = [covd[str(la)]["coverage_ori_pct"] for la in las]
    test_cov_pos = [covd[str(la)]["coverage_pos_pct"] for la in las]
    ci_lo_ori = [covd[str(la)]["ci_ori"][0] for la in las]
    ci_hi_ori = [covd[str(la)]["ci_ori"][1] for la in las]
    ci_lo_pos = [covd[str(la)]["ci_pos"][0] for la in las]
    ci_hi_pos = [covd[str(la)]["ci_pos"][1] for la in las]

    # Calibration constants - dual axis
    ax = axes[0]
    ax2 = ax.twinx()
    ln1 = ax.plot(las, qo, "o-", color=COLOR_CONFORMAL,
                    label="$\\hat{q}_{\\mathrm{ori}}$ (deg)",
                    markersize=5, linewidth=1.5)
    ln2 = ax2.plot(las, qp, "s--", color=COLOR_BASELINE,
                     label="$\\hat{q}_{\\mathrm{pos}}$ (cm)",
                     markersize=4.5, linewidth=1.5)
    ax.set_xlabel("Lookahead horizon (s)", fontsize=10)
    ax.set_ylabel("Orientation bound (deg)", fontsize=10, color=COLOR_CONFORMAL)
    ax2.set_ylabel("Position bound (cm)", fontsize=10, color=COLOR_BASELINE)
    ax.tick_params(axis="both", labelsize=9)
    ax.tick_params(axis="y", colors=COLOR_CONFORMAL)
    ax2.tick_params(axis="y", colors=COLOR_BASELINE, labelsize=9)
    ax.set_title("(a) Calibrated 90% bounds", fontsize=11, fontweight="bold", pad=10)
    lns = ln1 + ln2
    ax_a, lns_a = ax, lns
    ax2.grid(False)

    # Empirical coverage
    ax = axes[1]
    ax.axhline(90, ls="--", c="gray", alpha=0.8, linewidth=1.5,
                label="Target (90%)")

    # trajectory-level cluster-bootstrap intervals (from the calibration run)
    ci_ori = np.array([[c - lo for c, lo in zip(test_cov_ori, ci_lo_ori)],
                        [hi - c for c, hi in zip(test_cov_ori, ci_hi_ori)]])
    ci_pos = np.array([[c - lo for c, lo in zip(test_cov_pos, ci_lo_pos)],
                        [hi - c for c, hi in zip(test_cov_pos, ci_hi_pos)]])

    ax.errorbar(las, test_cov_ori, yerr=ci_ori, fmt="o-", color=COLOR_CONFORMAL,
                  markersize=5, linewidth=1.5, capsize=5,
                  label="Coverage (orientation)")
    ax.errorbar(las, test_cov_pos, yerr=ci_pos, fmt="s--", color=COLOR_BASELINE,
                  markersize=4.5, linewidth=1.5, capsize=5,
                  label="Coverage (position)")

    ax.set_xlabel("Lookahead horizon (s)", fontsize=10)
    ax.set_ylabel("Empirical test coverage (%)", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.set_ylim(84, 95)
    ax.set_title("(b) Coverage validation", fontsize=11, fontweight="bold", pad=10)
    ax_b = ax

    max_gap = max(abs(c - 90) for c in test_cov_ori + test_cov_pos)
    ax.text(0.5, 85.0, f"All within ±{max_gap:.1f}pp of target",
            fontsize=9, style="italic", color="gray")

    plt.tight_layout()
    ax_a.legend(lns_a, [l.get_label() for l in lns_a], loc="upper center",
                bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=True, fontsize=9)
    legend_below(ax_b, ncol=2, fontsize=9)
    plt.savefig(f"{OUT_DIR}/figure02_conformal_calibration.pdf")
    plt.savefig(f"{OUT_DIR}/figure02_conformal_calibration.png")
    plt.close()
    print("  -> figure02_conformal_calibration.{pdf,png}")

    # -----------------------------------------------------------------------

    # FIGURE 3: CDFs
    # -----------------------------------------------------------------------
    print("\nGenerating Figure 3: Error CDFs...")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2))

    # Position CDF
    ax = axes[0]
    for label, d, color in [("Baseline", phase1, COLOR_BASELINE),
                                  ("Conformal", conf, COLOR_CONFORMAL)]:
        sorted_v = np.sort(d["pos_tf"])
        # Normalize by ALL runs: the diverged run counts as never-captured,
        # so the conformal curve tops out at 134/135 = 99.3%.
        cdf = np.arange(1, len(sorted_v) + 1) / d["n_total"] * 100
        ax.plot(sorted_v, cdf, label=label, color=color, linewidth=1.3)
    ax.axvline(10, ls="--", c="gray", alpha=0.6,
                label="Capture threshold (10 cm)")
    p1_at_10 = np.sum(phase1["pos_tf"] < 10) / phase1["n_total"] * 100
    cf_at_10 = np.sum(conf["pos_tf"] < 10) / conf["n_total"] * 100
    p1_at_5 = np.sum(phase1["pos_tf"] < 5) / phase1["n_total"] * 100
    cf_at_5 = np.sum(conf["pos_tf"] < 5) / conf["n_total"] * 100
    # Annotate at 10 cm threshold
    ax.annotate(f"Baseline: {p1_at_10:.0f}%\nConformal: {cf_at_10:.0f}%",
                  xy=(10, 90), xytext=(13, 70),
                  fontsize=9, ha="left",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="gray", alpha=0.85),
                  arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5))
    # Annotate at 5 cm strict threshold (where conformal really helps)
    ax.annotate(f"At 5 cm:\nBaseline: {p1_at_5:.0f}%\nConformal: {cf_at_5:.0f}%",
                  xy=(5, p1_at_5), xytext=(6, 30),
                  fontsize=9, ha="left",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=COLOR_CONFORMAL, alpha=0.85),
                  arrowprops=dict(arrowstyle="->", color=COLOR_CONFORMAL, alpha=0.6))
    ax.set_xlabel("Position error at rendezvous (cm)")
    ax.set_ylabel("Cumulative % of runs")
    ax.set_title("(a) Position accuracy")
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 100)

    # Orientation CDF
    ax = axes[1]
    for label, d, color in [("Baseline", phase1, COLOR_BASELINE),
                                  ("Conformal", conf, COLOR_CONFORMAL)]:
        sorted_v = np.sort(d["ori_tf"])
        cdf = np.arange(1, len(sorted_v) + 1) / d["n_total"] * 100
        ax.plot(sorted_v, cdf, label=label, color=color, linewidth=1.3)
    ax.axvline(15, ls="--", c="gray", alpha=0.6,
                label="Capture threshold (15°)")
    p1_at_15 = np.sum(phase1["ori_tf"] < 15) / phase1["n_total"] * 100
    cf_at_15 = np.sum(conf["ori_tf"] < 15) / conf["n_total"] * 100
    p1_at_5 = np.sum(phase1["ori_tf"] < 5) / phase1["n_total"] * 100
    cf_at_5 = np.sum(conf["ori_tf"] < 5) / conf["n_total"] * 100
    ax.annotate(f"Baseline: {p1_at_15:.0f}%\nConformal: {cf_at_15:.0f}%",
                  xy=(15, 83), xytext=(22, 65),
                  fontsize=9, ha="left",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="gray", alpha=0.85),
                  arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5))
    ax.annotate(f"At 5°:\nBaseline: {p1_at_5:.0f}%\nConformal: {cf_at_5:.0f}%",
                  xy=(5, p1_at_5), xytext=(8, 30),
                  fontsize=9, ha="left",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=COLOR_CONFORMAL, alpha=0.85),
                  arrowprops=dict(arrowstyle="->", color=COLOR_CONFORMAL, alpha=0.6))
    ax.set_xlabel("Orientation error at rendezvous (deg)")
    ax.set_ylabel("Cumulative % of runs")
    ax.set_title("(b) Orientation accuracy")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 100)

    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    handles, labels = list(h0), list(l0)
    for hi, li in zip(h1, l1):
        if li not in labels:
            handles.append(hi); labels.append(li)
    plt.tight_layout()
    fig_legend_below(fig, handles, labels, ncol=4, fontsize=8)
    plt.savefig(f"{OUT_DIR}/figure03_error_cdfs.pdf")
    plt.savefig(f"{OUT_DIR}/figure03_error_cdfs.png")
    plt.close()
    print("  -> figure03_error_cdfs.{pdf,png}")

    # -----------------------------------------------------------------------

    # FIGURE 4: Per-tumble box plots
    # -----------------------------------------------------------------------
    print("\nGenerating Figure 4: Per-tumble box plots...")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.8))
    bin_labels = ["Low\n(<3 deg/s)", "Mid\n(3-10)", "High\n(10-20)",
                    "Extreme\n(>20)"]

    def bin_data(d):
        out = {b: [] for b in ["Low", "Mid", "High", "Extreme"]}
        for i in range(len(d["omega"])):
            o = d["omega"][i]
            if o < 3: out["Low"].append((d["pos_tf"][i], d["ori_tf"][i]))
            elif o < 10: out["Mid"].append((d["pos_tf"][i], d["ori_tf"][i]))
            elif o < 20: out["High"].append((d["pos_tf"][i], d["ori_tf"][i]))
            else: out["Extreme"].append((d["pos_tf"][i], d["ori_tf"][i]))
        return out

    p1_bins = bin_data(phase1)
    cf_bins = bin_data(conf)
    N_per_bin = [len(p1_bins[k]) for k in ["Low", "Mid", "High", "Extreme"]]
    bin_labels_with_n = [f"{lbl}\nn={n}" for lbl, n in zip(bin_labels, N_per_bin)]
    x = np.arange(4)
    w = 0.35

    def draw_box(ax, positions, data, color):
        bp = ax.boxplot(data, positions=positions, widths=w, patch_artist=True,
                          showfliers=True,
                          medianprops=dict(color="black", linewidth=1.3),
                          flierprops=dict(marker="o", markerfacecolor=color,
                                            markersize=3, markeredgecolor="none",
                                            alpha=0.5))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
            patch.set_alpha(0.7)
        return bp

    legend_elements = [Patch(facecolor=COLOR_BASELINE, edgecolor="black",
                                alpha=0.7, label="Baseline"),
                        Patch(facecolor=COLOR_CONFORMAL, edgecolor="black",
                                alpha=0.7, label="Conformal")]

    # Position
    ax = axes[0]
    p1_pos = [[v[0] for v in p1_bins[k]]
              for k in ["Low", "Mid", "High", "Extreme"]]
    cf_pos = [[v[0] for v in cf_bins[k]]
              for k in ["Low", "Mid", "High", "Extreme"]]
    draw_box(ax, x - w/2, p1_pos, COLOR_BASELINE)
    draw_box(ax, x + w/2, cf_pos, COLOR_CONFORMAL)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels_with_n, fontsize=9)
    ax.set_ylabel("Position error (cm)", fontsize=10)
    ax.set_title("(a) Position by tumble rate", fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(axis="y", labelsize=9)
    POS_CAP = 20.0
    ax.set_ylim(0, POS_CAP)
    for i in range(4):
        for vals, color, dx in [(p1_pos[i], COLOR_BASELINE, -w / 2),
                                  (cf_pos[i], COLOR_CONFORMAL, w / 2)]:
            k = sum(1 for v in vals if v > POS_CAP)
            if k:
                ax.text(x[i] + dx, POS_CAP * 0.99,
                          f"$\\blacktriangle$\u2009{k}",
                          ha="center", va="top", fontsize=9, color=color,
                          fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.15",
                                      facecolor="white", edgecolor="none",
                                      alpha=0.75))
    for i in range(4):
        if len(p1_pos[i]) > 0 and len(cf_pos[i]) > 0:
            med1 = np.median(p1_pos[i]); med2 = np.median(cf_pos[i])
            change = (med2 - med1) / med1 * 100 if med1 > 0 else 0
            if abs(change) > 5:
                color = "green" if change < 0 else "red"
                ax.text(i, ax.get_ylim()[1] * 0.85, f"{change:+.0f}%",
                          ha="center", fontsize=10, color=color,
                          fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.15",
                                      facecolor="white", edgecolor="none",
                                      alpha=0.75))

    # Orientation
    ax = axes[1]
    p1_ori = [[v[1] for v in p1_bins[k]]
              for k in ["Low", "Mid", "High", "Extreme"]]
    cf_ori = [[v[1] for v in cf_bins[k]]
              for k in ["Low", "Mid", "High", "Extreme"]]
    draw_box(ax, x - w/2, p1_ori, COLOR_BASELINE)
    draw_box(ax, x + w/2, cf_ori, COLOR_CONFORMAL)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels_with_n, fontsize=9)
    ax.set_ylabel("Orientation error (deg)", fontsize=10)
    ax.set_title("(b) Orientation by tumble rate", fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(axis="y", labelsize=9)
    ORI_CAP = 40.0
    ax.set_ylim(0, ORI_CAP)
    for i in range(4):
        for vals, color, dx in [(p1_ori[i], COLOR_BASELINE, -w / 2),
                                  (cf_ori[i], COLOR_CONFORMAL, w / 2)]:
            k = sum(1 for v in vals if v > ORI_CAP)
            if k:
                ax.text(x[i] + dx, ORI_CAP * 0.99,
                          f"$\\blacktriangle$\u2009{k}",
                          ha="center", va="top", fontsize=9, color=color,
                          fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.15",
                                      facecolor="white", edgecolor="none",
                                      alpha=0.75))
    for i in range(4):
        if len(p1_ori[i]) > 0 and len(cf_ori[i]) > 0:
            med1 = np.median(p1_ori[i]); med2 = np.median(cf_ori[i])
            change = (med2 - med1) / med1 * 100 if med1 > 0 else 0
            if abs(change) > 5:
                color = "green" if change < 0 else "red"
                ax.text(i, ax.get_ylim()[1] * 0.85, f"{change:+.0f}%",
                          ha="center", fontsize=10, color=color,
                          fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.15",
                                      facecolor="white", edgecolor="none",
                                      alpha=0.75))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig_legend_below(fig, legend_elements,
                     [h.get_label() for h in legend_elements],
                     ncol=2, fontsize=9)
    plt.savefig(f"{OUT_DIR}/figure04_per_tumble_breakdown.pdf")
    plt.savefig(f"{OUT_DIR}/figure04_per_tumble_breakdown.png")
    plt.close()
    print("  -> figure04_per_tumble_breakdown.{pdf,png}")

    # -----------------------------------------------------------------------

    # FIGURE 5: Capture scatter
    # -----------------------------------------------------------------------
    print("\nGenerating Figure 5: Capture scatter...")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.3))
    cr_baseline = ((phase1["pos_tf"] < 10) & (phase1["ori_tf"] < 15)).sum() \
        / phase1["n_total"] * 100
    cr_conformal = ((conf["pos_tf"] < 10) & (conf["ori_tf"] < 15)).sum() \
        / conf["n_total"] * 100


    for ax, data, title in [
            (axes[0], phase1, f"(a) Baseline\n{cr_baseline:.0f}% capture-ready"),
            (axes[1], conf, f"(b) Conformal\n{cr_conformal:.0f}% capture-ready")]:

        ax.scatter(data["pos_tf"], data["ori_tf"], c=data["omega"], cmap="viridis",
                     s=14, alpha=0.85, edgecolors="black", linewidths=0.4,
                     vmin=0, vmax=30)
        ax.axvline(10, ls="--", c="gray", alpha=0.5)
        ax.axhline(15, ls="--", c="gray", alpha=0.5)
        ax.fill_between([0, 10], 0, 15, color="green", alpha=0.07,
                          label="Capture-ready zone")
        ax.set_xlabel("Position error (cm)")
        ax.set_ylabel("Orientation error (deg)")
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 80)
        legend_below(ax, ncol=1, fontsize=9)

    sm = plt.cm.ScalarMappable(cmap="viridis",
                                  norm=plt.Normalize(vmin=0, vmax=30))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                          fraction=0.025, pad=0.02)
    cbar.set_label("Tumble rate (deg/s)", size=9)
    cbar.ax.tick_params(labelsize=8)
    plt.savefig(f"{OUT_DIR}/figure05_capture_scatter.pdf", bbox_inches="tight")
    plt.savefig(f"{OUT_DIR}/figure05_capture_scatter.png", bbox_inches="tight")
    plt.close()
    print("  -> figure05_capture_scatter.{pdf,png}")

    # -----------------------------------------------------------------------

    # FIGURE 6: Rejection-stratified capture analysis
    # -----------------------------------------------------------------------
    print("\nGenerating Figure 6: Rejection-stratified capture analysis...")
 
    # Stratify by NUMBER OF REJECTIONS (meaningful, varies across runs)
    # Hypothesis: 0 rejections = "easy" runs (UPN confident), high success
    #             1-3 rejections = "moderate" caution triggered
    #             4+ rejections = "difficult" runs (UPN uncertain often)
    strata = {
        "0 rejections": [],
        "1-3 rejections": [],
        "4+ rejections": [],
    }
    for i in range(len(conf["n_rejected"])):
        nr = conf["n_rejected"][i]
        pos = conf["pos_tf"][i]; ori = conf["ori_tf"][i]
        captured = (pos < 10) and (ori < 15)
        if nr == 0: strata["0 rejections"].append(captured)
        elif nr <= 3: strata["1-3 rejections"].append(captured)
        else: strata["4+ rejections"].append(captured)
 
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.3))
 
    # Panel (a): capture rate per rejection stratum
    ax = axes[0]
    group_names = list(strata.keys())
    capture_rates = [np.mean(strata[g]) * 100 if len(strata[g]) > 0 else 0
                       for g in group_names]
    group_ns = [len(strata[g]) for g in group_names]
    ax.bar(range(len(group_names)), capture_rates,
            color=["#2ca02c", "#ff7f0e", "#d62728"], alpha=0.8,
            edgecolor="black", linewidth=1.0)
    ax.set_xticks(range(len(group_names)))
    ax.set_xticklabels(group_names, fontsize=9)
    ax.set_ylabel("Capture-ready rate (%)", fontsize=10)
    ax.set_title("(a) Capture rate by rejection count",
                 fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylim(0, 105)
    cr_all = ((conf["pos_tf"] < 10) & (conf["ori_tf"] < 15)).sum() \
        / conf["n_total"] * 100
    ax.axhline(cr_all, color="gray", ls=":", alpha=0.7, linewidth=1.5,
               label=f"Overall: {cr_all:.0f}%")
    ax_bar = ax
    for i, (rate, n) in enumerate(zip(capture_rates, group_ns)):
        ax.text(i, rate + 2, f"{rate:.0f}%", ha="center",
                  fontsize=10, fontweight="bold")
        ax.text(i, 5, f"n={n}", ha="center", fontsize=9, color="white")
 
    # Panel (b): rejection frequency histogram (unchanged)
    ax = axes[1]
    counts, bins, _ = ax.hist(conf["n_rejected"],
                                  bins=range(0, max(conf["n_rejected"]) + 2),
                                  color=COLOR_CONFORMAL, edgecolor="black",
                                  linewidth=0.7, alpha=0.85)
    ax.set_xlabel("Number of trajectory rejections per run", fontsize=10)
    ax.set_ylabel("Number of runs", fontsize=10)
    ax.set_title("(b) Rejection frequency", fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(axis="both", labelsize=9)
    mean_rej = np.mean(conf["n_rejected"])
    ax.axvline(mean_rej, color="black", ls="--", alpha=0.6, linewidth=1.5,
                label=f"Mean: {mean_rej:.1f} rejections")
    fraction_under_5 = sum(1 for r in conf["n_rejected"] if r <= 5) / len(conf["n_rejected"]) * 100
    ax.text(2.5, max(counts) * 0.85,
             f"{fraction_under_5:.0f}% of runs:\n≤5 rejections",
             fontsize=9, ha="left",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                         edgecolor="gray", alpha=0.7))
    outlier_rej = [r for r in conf["n_rejected"] if r > 10]
    if outlier_rej:
        ax.annotate("Persistent\nfailure cases",
                      xy=(max(outlier_rej), 1),
                      xytext=(max(outlier_rej) - 5, 8),
                      fontsize=9, color="red",
                      arrowprops=dict(arrowstyle="->",
                                          color="red", alpha=0.7, lw=1.5))
    ax_hist = ax

    plt.tight_layout()
    legend_below(ax_bar, ncol=1)
    legend_below(ax_hist, ncol=1)
    plt.savefig(f"{OUT_DIR}/figure06_rejection_stratified.pdf")
    plt.savefig(f"{OUT_DIR}/figure06_rejection_stratified.png")
    plt.close()
    print("  -> figure06_rejection_stratified.{pdf,png}")
 
    for f in ["fig5_threshold_sweep", "fig9_orientation_time", "fig6_rejection_stats"]:
        for ext in ["pdf", "png"]:
            p = f"{OUT_DIR}/{f}.{ext}"
            if os.path.exists(p):
                os.remove(p)
                print(f"  Removed deprecated {p}")


def generate_tables(phase1, conf, calib):
    """Tables 1-4 in LaTeX."""
    print("\nGenerating tables...")

    # Table 1
    lines = [r"% Table 1: Overall results comparison",
             r"\begin{table}[t]", r"\centering",
             r"\caption{Overall rendezvous performance across 45 trajectories $\times$ 3 seeds = 135 runs. All 135 runs of both configurations complete; statistics are computed over all 135 runs.}",
             r"\label{tab:overall}",
             r"\begin{tabular}{lcc}", r"\toprule",
             r"\textbf{Metric} & \textbf{Baseline} & \textbf{Conformal} \\",
             r"\midrule"]
    for label, key, fmt in [
        ("Position mean (at $t_f$)", "pos_tf", ".2f"),
        ("Position median", "pos_tf", ".2f"),
        ("Position 95th percentile", "pos_tf", ".2f"),
        ("Orientation mean (at $t_f$)", "ori_tf", ".2f"),
        ("Orientation median", "ori_tf", ".2f"),
        ("Orientation 95th percentile", "ori_tf", ".2f")]:
        if "median" in label:
            v1 = float(np.median(phase1[key])); v2 = float(np.median(conf[key]))
        elif "95th" in label:
            v1 = float(np.percentile(phase1[key], 95))
            v2 = float(np.percentile(conf[key], 95))
        else:
            v1 = float(np.mean(phase1[key])); v2 = float(np.mean(conf[key]))
        unit_tex = r"\,cm" if "Position" in label else r"$^{\circ}$"
        lines.append(f"{label} & ${v1:{fmt}}${unit_tex} & ${v2:{fmt}}${unit_tex} \\\\")
    cr1 = ((phase1["pos_tf"] < 10) & (phase1["ori_tf"] < 15)).sum() \
        / phase1["n_total"] * 100
    cr2 = ((conf["pos_tf"] < 10) & (conf["ori_tf"] < 15)).sum() \
        / conf["n_total"] * 100
    lines.append(r"\midrule")
    lines.append(f"Capture-ready rate & ${cr1:.0f}\\%$ & ${cr2:.0f}\\%$ \\\\")
    s1 = ((phase1['pos_tf'] < 5) & (phase1['ori_tf'] < 5)).sum() \
        / phase1['n_total'] * 100
    s2 = ((conf['pos_tf'] < 5) & (conf['ori_tf'] < 5)).sum() \
        / conf['n_total'] * 100
    lines.append(f"Strict capture (5\\,cm \\& 5$^{{\\circ}}$) & ${s1:.0f}\\%$ & ${s2:.0f}\\%$ \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(f"{OUT_DIR}/table2_overall_results.tex", "w") as f:
        f.write("\n".join(lines))
    print("  -> table2_overall_results.tex")

    # Table 2 (calibration)
    lines = [r"% Table 2: Conformal calibration constants and coverage",
             r"\begin{table}[t]", r"\centering",
             r"\caption{Conformal calibration constants $\hat{q}$ at 90\% confidence, with empirical coverage measured on the full held-out test set (45 trajectories). Bounds use the hierarchical repeated-subsampling construction over 200 independently generated calibration trajectories.}",
             r"\label{tab:calibration}",
             r"\begin{tabular}{cccccc}", r"\toprule",
             r"\textbf{Lookahead} & $n$ & $\hat{q}_{\mathrm{ori}}$ & $\hat{q}_{\mathrm{pos}}$ & \textbf{Cov.\ ori} & \textbf{Cov.\ pos} \\",
             r" & & (deg) & (cm) & (\%) & (\%) \\", r"\midrule"]
    covd = load_test_coverage()
    test_cov = {la: (covd[la]["coverage_ori_pct"], covd[la]["coverage_pos_pct"])
                for la in ["0.5", "1.0", "1.5", "2.0"]}
    for la in ["0.5", "1.0", "1.5", "2.0"]:
        n = calib["calibration_n_samples"][la]
        qo = calib["q_hat_orientation_deg"][la]
        qp = calib["q_hat_position_cm"][la]
        co, cp = test_cov[la]
        lines.append(f"${la}$\\,s & {n} & ${qo:.2f}$ & ${qp:.2f}$ & ${co:.1f}$ & ${cp:.1f}$ \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(f"{OUT_DIR}/table1_calibration_constants.tex", "w") as f:
        f.write("\n".join(lines))
    print("  -> table1_calibration_constants.tex")

    # Table 3 (per-tumble, median+IQR)
    lines = [r"% Table 3: Per-tumble breakdown (median + IQR)",
             r"\begin{table}[t]", r"\centering",
             r"\caption{Per-tumble-rate breakdown of rendezvous accuracy. Median (interquartile range).}",
             r"\label{tab:per_tumble}",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"\textbf{Bin} & $N$ & \textbf{Pos baseline} & \textbf{Pos conformal} & \textbf{Ori baseline} & \textbf{Ori conformal} \\",
             r" & & (cm) & (cm) & (deg) & (deg) \\", r"\midrule"]
    bins = ["Low ($<3$\\,deg/s)", "Mid (3-10)", "High (10-20)", "Extreme ($>20$)"]
    bd = {b: {"phase1_pos": [], "phase1_ori": [], "conf_pos": [], "conf_ori": []}
          for b in bins}

    def bin_label(o):
        if o < 3: return bins[0]
        elif o < 10: return bins[1]
        elif o < 20: return bins[2]
        else: return bins[3]
    for i in range(len(phase1["omega"])):
        b = bin_label(phase1["omega"][i])
        bd[b]["phase1_pos"].append(phase1["pos_tf"][i])
        bd[b]["phase1_ori"].append(phase1["ori_tf"][i])
    for i in range(len(conf["omega"])):
        b = bin_label(conf["omega"][i])
        bd[b]["conf_pos"].append(conf["pos_tf"][i])
        bd[b]["conf_ori"].append(conf["ori_tf"][i])
    for b in bins:
        n = len(bd[b]["phase1_pos"])
        if n == 0: continue
        p1m = np.median(bd[b]["phase1_pos"])
        p1q = np.percentile(bd[b]["phase1_pos"], [25, 75])
        p2m = np.median(bd[b]["conf_pos"])
        p2q = np.percentile(bd[b]["conf_pos"], [25, 75])
        o1m = np.median(bd[b]["phase1_ori"])
        o1q = np.percentile(bd[b]["phase1_ori"], [25, 75])
        o2m = np.median(bd[b]["conf_ori"])
        o2q = np.percentile(bd[b]["conf_ori"], [25, 75])
        lines.append(f"{b} & {n} & ${p1m:.2f}$ ({p1q[0]:.2f}-{p1q[1]:.2f}) & ${p2m:.2f}$ ({p2q[0]:.2f}-{p2q[1]:.2f}) & ${o1m:.2f}$ ({o1q[0]:.2f}-{o1q[1]:.2f}) & ${o2m:.2f}$ ({o2q[0]:.2f}-{o2q[1]:.2f}) \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(f"{OUT_DIR}/table3_per_tumble.tex", "w") as f:
        f.write("\n".join(lines))
    print("  -> table3_per_tumble.tex")

    # Table 4 (capture thresholds)
    lines = [r"% Table 4: Joint capture-ready rates",
             r"\begin{table}[t]", r"\centering",
             r"\caption{Joint capture-ready success rates at multiple thresholds.}",
             r"\label{tab:thresholds}",
             r"\begin{tabular}{ccc c}", r"\toprule",
             r"\textbf{Pos thr.} & \textbf{Ori thr.} & \textbf{Baseline} & \textbf{Conformal} \\",
             r" (cm) & (deg) & (\%) & (\%) \\", r"\midrule"]
    for p_th, o_th in [(5, 5), (5, 10), (10, 10), (10, 15), (15, 15), (15, 20)]:
        n1 = ((phase1["pos_tf"] < p_th) & (phase1["ori_tf"] < o_th)).sum() \
            / phase1["n_total"] * 100
        n2 = ((conf["pos_tf"] < p_th) & (conf["ori_tf"] < o_th)).sum() \
            / conf["n_total"] * 100
        lines.append(f"{p_th} & {o_th} & ${n1:.0f}$ & ${n2:.0f}$ \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(f"{OUT_DIR}/table4_capture_thresholds.tex", "w") as f:
        f.write("\n".join(lines))
    print("  -> table4_capture_thresholds.tex")


# ============================================================================
# Trajectory figures (require live simulation)
# ============================================================================
def generate_trajectory_figures(calib):
    """Figures 7, 8, 10, 11 - require running cap_control simulation."""
    print("\nImporting cap_control modules for live simulation...")

    # Clear cached imports
    for m in list(sys.modules.keys()):
        if "cap_control" in m or "space_robot_dq" in m:
            del sys.modules[m]

    from cap_control import config as cfg
    from cap_control.dynamics.free_floating import FreeFloatingChaser
    from cap_control.controller.feedback_linearization import FeedbackLinearizationController
    from cap_control.control.rendezvous_trajectory import (
        solve_rendezvous_trajectory, _grapple_kinematics,
        _trajectory_coefficients, RendezvousTrajectory)
    from cap_control.simulation.target_sim import DatasetTumblingTarget
    from cap_control.simulation.sensors import NoisyPoseSensor
    from cap_control.prediction.upn_predictor import UPNPredictor

    # Conformal helpers
    CALIB_LA = calib["lookaheads"]
    CALIB_Q_ORI = [calib["q_hat_orientation_deg"][str(la)] for la in CALIB_LA]
    CALIB_Q_POS = [calib["q_hat_position_cm"][str(la)] for la in CALIB_LA]

    def q_hat_ori(la):
        if la <= CALIB_LA[0]: return CALIB_Q_ORI[0]
        if la >= CALIB_LA[-1]: return CALIB_Q_ORI[-1]
        return float(np.interp(la, CALIB_LA, CALIB_Q_ORI))

    def q_hat_pos(la):
        if la <= CALIB_LA[0]: return CALIB_Q_POS[0]
        if la >= CALIB_LA[-1]: return CALIB_Q_POS[-1]
        return float(np.interp(la, CALIB_LA, CALIB_Q_POS))

    RHO_BODY = np.array([0.1, 0.0, 0.0])
    IC = np.diag([10.0, 10.0, 10.0])

    taus_grid = np.linspace(0.5, 2.0, 301)
    TAU_CONTINGENCY = float(max([t for t in taus_grid if q_hat_ori(t) <= 40.0],
                                 default=0.5))

    def plan_fixed_horizon(rh, rhdot, prop, tf_fix, sigma=1.0):
        q_tf, omega_tf, r_tf, _ = prop(tf_fix)
        q_tf = q_tf / max(np.linalg.norm(q_tf), 1e-12)
        rc_tf, rcdot_tf, _ = _grapple_kinematics(q_tf, omega_tf, r_tf,
                                                   RHO_BODY, IC)
        k0, k1, k2, k3 = _trajectory_coefficients(rh, rhdot, rc_tf, rcdot_tf,
                                                    tf_fix, sigma)
        return RendezvousTrajectory(k0=k0, k1=k1, k2=k2, k3=k3, sigma=sigma,
                                      tf=tf_fix, rc_tf=rc_tf,
                                      rcdot_tf=rcdot_tf)

    def make_upn_propagator(upn, obs_h, obs_t, t_offset):
        HIST = cfg.UPN_HISTORY_LEN
        def prop(t):
            ho = np.asarray(obs_h[-HIST:])
            ht = np.asarray(obs_t[-HIST:])
            ft = np.array([max(t_offset + t, ht[-1] + 1e-3)])
            m, _, _ = upn.predict(ho, ht, ft, future_obs=None,
                                       use_updates=False)
            s = m[-1]
            q = s[6:10] / max(np.linalg.norm(s[6:10]), 1e-12)
            return (q, s[10:13], s[0:3], s[3:6])
        return prop

    def run_with_logging(traj_idx, seed=1, t_final=4.0, dt=0.1,
                           kp_ori_max=5.0, t_blend=1.5,
                           reject_threshold=40.0):
        chaser = FreeFloatingChaser()
        state = chaser.home()
        target_sim = DatasetTumblingTarget(traj_idx=traj_idx)
        sensor = NoisyPoseSensor(pos_noise_std=cfg.POS_NOISE_STD,
                                 rot_noise_std_deg=cfg.ROT_NOISE_STD_DEG, seed=seed)
        upn = UPNPredictor()

        HIST = cfg.UPN_HISTORY_LEN; SDT = cfg.SENSOR_DT
        obs_h, obs_t = [], []
        for tb in np.arange(HIST) * SDT - (HIST - 1) * SDT:
            p, q = target_sim.pose_at(max(0.0, tb))
            obs_h.append(sensor.observe(p, q)); obs_t.append(float(tb))
        next_sensor_t = 0.0

        rh, _ = chaser.fk_world(state)
        temp_c = FeedbackLinearizationController(
            chaser=chaser, Kp_pos=20.0, Kd_pos=8.0,
            Kp_ori=kp_ori_max, Kd_ori=kp_ori_max * 0.8,
            tau_limit=10.0, t_blend_ori=t_blend)
        rhdot = temp_c.compute_ee_velocity(state)
        prop = make_upn_propagator(upn, obs_h, obs_t, t_offset=0.0)
        traj = solve_rendezvous_trajectory(
            rh, rhdot, target_state=None, target_inertia=IC,
            rho_body=RHO_BODY, w1=1.0, w2=1.0, target_propagator=prop)
        plan_t0 = 0.0

        initial_q_hat = q_hat_ori(traj.tf)
        current_kp_ori = kp_ori_from_uncertainty(initial_q_hat)
        consec_rej = 0
        controller = FeedbackLinearizationController(
            chaser=chaser, Kp_pos=20.0, Kd_pos=8.0,
            Kp_ori=current_kp_ori, Kd_ori=current_kp_ori * 0.8,
            tau_limit=10.0, t_blend_ori=t_blend)

        n_steps = int(round(t_final / dt))
        log = {"t": [], "ee_pos": [], "ee_quat": [], "target_pos": [],
                "target_quat": [], "grasp_pos": [], "des_pos": [], "des_quat": [],
                "pos_err": [], "ori_err": [], "q_hat_ori_at_lookahead": [],
                "q_hat_pos_at_lookahead": [], "kp_ori_active": [],
                "tf": [], "replan_events": [], "rejected_events": []}

        for k in range(n_steps):
            t_now = k * dt
            while next_sensor_t <= t_now + 1e-9:
                p, q = target_sim.pose_at(next_sensor_t)
                obs_h.append(sensor.observe(p, q)); obs_t.append(next_sensor_t)
                if len(obs_h) > HIST + 10:
                    obs_h = obs_h[-(HIST + 10):]; obs_t = obs_t[-(HIST + 10):]
                next_sensor_t += SDT

            time_in = t_now - plan_t0
            if time_in >= 1.0 or time_in >= traj.tf - dt:
                rh, _ = chaser.fk_world(state)
                rhdot = controller.compute_ee_velocity(state)
                prop = make_upn_propagator(upn, obs_h, obs_t, t_offset=t_now)
                try:
                    new_traj = solve_rendezvous_trajectory(
                        rh, rhdot, target_state=None, target_inertia=IC,
                        rho_body=RHO_BODY, w1=1.0, w2=1.0, target_propagator=prop)
                    nq = q_hat_ori(new_traj.tf)
                    if nq > reject_threshold and consec_rej < 4:
                        log["rejected_events"].append(t_now)
                        consec_rej += 1
                    else:
                        if nq > reject_threshold:
                            # Mechanism 3: short fixed-horizon contingency plan
                            log["rejected_events"].append(t_now)
                            new_traj = plan_fixed_horizon(rh, rhdot, prop,
                                                            TAU_CONTINGENCY)
                            nq = q_hat_ori(TAU_CONTINGENCY)
                        consec_rej = 0
                        traj = new_traj
                        plan_t0 = t_now
                        log["replan_events"].append(t_now)
                        current_kp_ori = kp_ori_from_uncertainty(nq)
                        controller = FeedbackLinearizationController(
                            chaser=chaser, Kp_pos=20.0, Kd_pos=8.0,
                            Kp_ori=current_kp_ori,
                            Kd_ori=current_kp_ori * 0.8,
                            tau_limit=10.0, t_blend_ori=t_blend)
                except Exception:
                    pass

            local_t = t_now - plan_t0
            rh_des, rhdot_des, rhddot_des = traj.evaluate(local_t)
            ref = {"rh_des": rh_des, "rhdot_des": rhdot_des,
                    "rhddot_des": rhddot_des}

            time_to_go = traj.tf - local_t
            future_t = t_now + time_to_go
            ho = np.asarray(obs_h[-HIST:])
            ht = np.asarray(obs_t[-HIST:])
            ft = np.array([max(future_t, ht[-1] + 1e-3)])
            try:
                m, _, _ = upn.predict(ho, ht, ft, future_obs=None,
                                          use_updates=False)
                s_pred = m[-1]
                q_des = s_pred[6:10] / max(np.linalg.norm(s_pred[6:10]), 1e-12)
                ref["q_des"] = q_des
                ref["omega_des"] = s_pred[10:13]
                ref["omega_dot_des"] = np.zeros(3)
                ref["time_to_go"] = float(time_to_go)
            except Exception:
                pass

            tau, _ = controller.solve(state, ref)
            state = chaser.dynamic_step(state, tau, dt, include_coriolis=True)

            rh_actual, q_ee = chaser.fk_world(state)
            true_st = target_sim.state_at(t_now + dt)  # aligned with post-step chaser
            q_true = true_st[6:10] / max(np.linalg.norm(true_st[6:10]), 1e-12)
            rc_actual, _, _ = _grapple_kinematics(q_true, true_st[10:13],
                                                          true_st[0:3], RHO_BODY, IC)

            log["t"].append(t_now)
            log["ee_pos"].append(rh_actual.copy())
            log["ee_quat"].append(q_ee.copy())
            log["target_pos"].append(true_st[0:3].copy())
            log["target_quat"].append(q_true.copy())
            log["grasp_pos"].append(rc_actual.copy())
            log["des_pos"].append(rh_des.copy())
            log["des_quat"].append(ref.get("q_des", q_true).copy())
            log["pos_err"].append(np.linalg.norm(rh_actual - rc_actual) * 100)
            log["ori_err"].append(quat_angle_deg(q_ee, q_true))
            log["q_hat_ori_at_lookahead"].append(q_hat_ori(time_to_go))
            log["q_hat_pos_at_lookahead"].append(q_hat_pos(time_to_go))
            log["kp_ori_active"].append(current_kp_ori)
            log["tf"].append(traj.tf - local_t)

        for k in log:
            log[k] = np.array(log[k])
        return log

    # Run two scenarios
    print("\nRecording SUCCESS run (traj 13, seed 2)...")
    log_success = run_with_logging(traj_idx=13, seed=2)
    print(f"  Final: {log_success['pos_err'][-1]:.2f} cm / {log_success['ori_err'][-1]:.2f} deg")

    print("\nRecording FAILURE run (traj 33, seed 1)...")
    log_failure = run_with_logging(traj_idx=33, seed=1)
    print(f"  Final: {log_failure['pos_err'][-1]:.2f} cm / {log_failure['ori_err'][-1]:.2f} deg")

    # FIGURE 7: 3D trajectory
    print("\nGenerating Figure 7: 3D trajectory...")
    fig = plt.figure(figsize=(7.16, 4.5))

    for idx, (log, title) in enumerate(
        [(log_success, "(a) Successful capture (traj 13)"),
          (log_failure, "(b) Failure case (traj 33)")]):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        ee = log["ee_pos"]; grasp = log["grasp_pos"]

        # Main trajectories — clean solid lines
        ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], color=COLOR_EE,
                linewidth=2.0, alpha=0.9, solid_capstyle="round",
                label="End-effector")
        ax.plot(grasp[:, 0], grasp[:, 1], grasp[:, 2], color=COLOR_GRASP,
                linewidth=1.8, alpha=0.9, linestyle="--",
                label="Grasp point (target)")
        ax.scatter(ee[::10, 0], ee[::10, 1], ee[::10, 2],
                    color=COLOR_EE, s=9, alpha=0.55, depthshade=False)

        # Key markers
        ax.scatter(*ee[0], color=COLOR_EE, s=110, marker="o",
                    edgecolor="black", linewidth=1.5,
                    label="EE start", zorder=5, depthshade=False)
        ax.scatter(*ee[-1], color=COLOR_EE, s=170, marker="*",
                    edgecolor="black", linewidth=1.5,
                    label="EE end", zorder=5, depthshade=False)
        ax.scatter(*grasp[-1], color=COLOR_GRASP, s=170, marker="X",
                    edgecolor="black", linewidth=1.5,
                    label="Capture point", zorder=5, depthshade=False)

        # Gap annotation
        gap_cm = np.linalg.norm(ee[-1] - grasp[-1]) * 100
        mid = (ee[-1] + grasp[-1]) / 2
        ax.plot([ee[-1, 0], grasp[-1, 0]],
                [ee[-1, 1], grasp[-1, 1]],
                [ee[-1, 2], grasp[-1, 2]],
                'k--', linewidth=1.5, alpha=0.6)
        # (gap label drawn after view_init below, in screen-space offset)

        # Axes styling
        ax.set_xlabel("$x$ (m)", fontsize=10, labelpad=12)
        ax.set_ylabel("$y$ (m)", fontsize=10, labelpad=12)
        ax.set_zlabel("$z$ (m)", fontsize=10, labelpad=6, rotation=90)
        from matplotlib.ticker import MaxNLocator, FormatStrFormatter
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(3))
            axis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(axis="x", labelsize=8, pad=4)
        ax.tick_params(axis="y", labelsize=8, pad=4)
        ax.tick_params(axis="z", labelsize=8, pad=4)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

        elev, azim = [(25, 40), (20, 50)][idx]   # VIEWS: tweak here
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 0.85))
        ax.zaxis.set_rotate_label(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#CCCCCC')
        ax.yaxis.pane.set_edgecolor('#CCCCCC')
        ax.zaxis.pane.set_edgecolor('#CCCCCC')
        ax.grid(True, alpha=0.15, linewidth=0.5)

        from mpl_toolkits.mplot3d import proj3d
        px, py, _ = proj3d.proj_transform(mid[0], mid[1], mid[2],
                                            ax.get_proj())
        ax.annotate(f"{gap_cm:.1f} cm", xy=(px, py),
                    xytext=[(-12, 16), (12, 18)][idx],
                    textcoords="offset points",
                    fontsize=9, ha=["right", "left"][idx],
                    color="black", fontweight="bold",
                    zorder=20, annotation_clip=False,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                edgecolor="gray", alpha=0.9))

    # Shared legend below — one line, tight
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               fontsize=9, frameon=True, framealpha=0.95,
               bbox_to_anchor=(0.5, 0.01),
               handlelength=2.0, columnspacing=2.0,
               edgecolor="gray")

    plt.tight_layout(rect=[0, 0.07, 1, 0.97], w_pad=3.5)
    plt.savefig(f"{OUT_DIR}/figure07_trajectory_3d.pdf")
    plt.savefig(f"{OUT_DIR}/figure07_trajectory_3d.png")
    plt.close()
    print("  -> figure07_trajectory_3d.{pdf,png}")

    # FIGURE 8: per-axis tracking error
    print("\nGenerating Figure 8: per-axis tracking error...")
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 4.6), sharex=True)
    log = log_success
    t = log["t"]

    for i in range(3):
        ax = axes[i]
        err_axis = (log["ee_pos"][:, i] - log["grasp_pos"][:, i]) * 100
        ax.plot(t, err_axis, color=COLOR_EE, linewidth=1.5,
                 label=("EE - grasp point" if i == 0 else None))
        ax.axhline(0, color="black", linestyle="-", alpha=0.4, linewidth=0.8)
        ax.axhline(10, color="gray", linestyle="--", alpha=0.5, linewidth=1.2,
                    label=("±10 cm capture threshold" if i == 0 else None))
        ax.axhline(-10, color="gray", linestyle="--", alpha=0.5, linewidth=1.2)
        ax.set_ylabel(f"${['x','y','z'][i]}$ error (cm)", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)

        err_max = max(abs(err_axis).max(), 10)
        if i == 0:
            ax.set_ylim(-15, 30)
        elif i == 1:
            ax.set_ylim(-15, 15)
        else:
            ax.set_ylim(-err_max * 0.5, err_max * 1.25)

        if i == 0:
            ax.set_title("Per-axis position tracking error vs time\n(traj 13, seed 2)",
                         fontsize=9, fontweight="bold", pad=8)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    plt.tight_layout(h_pad=1.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig_legend_below(fig, handles, labels, ncol=1, fontsize=8)
    plt.savefig(f"{OUT_DIR}/figure08_position_tracking_error.pdf")
    plt.savefig(f"{OUT_DIR}/figure08_position_tracking_error.png")
    plt.close()
    print("  -> figure08_position_tracking_error.{pdf,png}")

    # FIGURE 9: tracking errors success vs failure
    print("\nGenerating Figure 9: tracking errors with thresholds...")
    POS_YMAX = max(np.max(log_success["pos_err"]),
                     np.max(log_failure["pos_err"]))
    ORI_YMAX = max(np.max(log_success["ori_err"]),
                     np.max(log_failure["ori_err"]))
    POS_YMAX = float(np.ceil(POS_YMAX / 10.0) * 10)
    ORI_YMAX = float(np.ceil(ORI_YMAX / 20.0) * 20)

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.8), sharex="col")
    for col_idx, (log, title) in enumerate(
        [(log_success, "Successful capture (traj 13)"),
          (log_failure, "Failure case (traj 33)")]):
        t = log["t"]
        rejs = log["rejected_events"]
        n_rejs = len(rejs)
        n_replans = len(log["replan_events"])
        rej_subtitle = (f"({n_replans} replan{'s' if n_replans != 1 else ''}, "
                        f"{n_rejs} rejection{'s' if n_rejs != 1 else ''})")

        # Position
        ax = axes[0, col_idx]
        ax.fill_between(t, 0, log["q_hat_pos_at_lookahead"],
                          color=COLOR_UNCERTAINTY, alpha=0.18,
                          label=("90% bound $\\hat{q}_{\\mathrm{pos}}$"
                                  if col_idx == 0 else None))
        ax.plot(t, log["pos_err"], color=COLOR_EE, linewidth=1.4,
                 label=("Tracking error" if col_idx == 0 else None))
        ax.axhline(10, color="green", linestyle="--", alpha=0.6, linewidth=1.0,
                    label=("Capture threshold (10 cm)" if col_idx == 0 else None))
        for ev in log["replan_events"][:3]:
            ax.axvline(ev, color="gray", linestyle=":", alpha=0.4)
        if n_rejs > 0:
            ax.axvline(rejs[0], color="red", linestyle=":", alpha=0.6,
                        linewidth=1.0)
        ax.set_ylabel("Position error (cm)")
        ax.set_ylim(0, POS_YMAX)
        ax.set_title(f"{title}\n{rej_subtitle}", fontsize=9)

        # Orientation
        ax = axes[1, col_idx]
        ax.fill_between(t, 0, log["q_hat_ori_at_lookahead"],
                          color=COLOR_UNCERTAINTY, alpha=0.18,
                          label=("90% bound $\\hat{q}_{\\mathrm{ori}}$"
                                  if col_idx == 0 else None))
        ax.plot(t, log["ori_err"], color=COLOR_EE, linewidth=1.4,
                 label=("Tracking error" if col_idx == 0 else None))
        ax.axhline(15, color="green", linestyle="--", alpha=0.6, linewidth=1.0,
                    label=("Capture threshold (15°)" if col_idx == 0 else None))
        for ev in log["replan_events"][:3]:
            ax.axvline(ev, color="gray", linestyle=":", alpha=0.4,
                        label=("Replan" if col_idx == 0
                                and ev == log["replan_events"][0] else None))
        if n_rejs > 0:
            ax.axvline(rejs[0], color="red", linestyle=":", alpha=0.6,
                        linewidth=1.0,
                        label=("Rejection" if col_idx == 0 else None))
        ax.set_ylabel("Orientation error (deg)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, ORI_YMAX)

    # Shared legend below (deduplicated across both rows of the first column)
    handles, labels = [], []
    for src_ax in (axes[0, 0], axes[1, 0]):
        h, l = src_ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi); labels.append(li)
    fig_legend_below(fig, handles, labels, ncol=4)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/figure09_tracking_errors_envelope.pdf")
    plt.savefig(f"{OUT_DIR}/figure09_tracking_errors_envelope.png")
    plt.close()
    print("  -> figure09_tracking_errors_envelope.{pdf,png}")

    # FIGURE 10: uncertainty management (2 panels)
    # FIGURE 10: uncertainty management (2 panels, per-panel legends)
    print("\nGenerating Figure 10: conformal uncertainty + gain modulation...")
    
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 5.85), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1.0]})
    log = log_success
    t = log["t"]
    t0, t1 = float(t[0]), float(t[-1])

    # Top: q_hat with three-zone shading
    ax = axes[0]
    ax.fill_between(t, 0, 20, color="green", alpha=0.07)
    ax.fill_between(t, 20, 40, color="orange", alpha=0.07)
    ax.fill_between(t, 40, 95, color="red", alpha=0.07)
    ax.plot(t, log["q_hat_ori_at_lookahead"], color=COLOR_UNCERTAINTY,
            linewidth=1.8, zorder=3,
            label="$\\hat{q}_{\\mathrm{ori}}(\\tau)$ at lookahead $\\tau$")
    ax.axhline(40, color="red", ls="--", alpha=0.7, linewidth=1.5,
               label="Reject threshold (40°)")
    ax.axhline(20, color="orange", ls="--", alpha=0.7, linewidth=1.5,
               label="Gain-low threshold (20°)")
    for y, txt, c in [(9.0, "full gain", "darkgreen"),
                      (33.0, "modulated", "darkorange"),
                      (66.0, "rejected", "darkred")]:
        ax.text(t1 - 0.05, y, txt, ha="right", va="center", fontsize=8,
                alpha=0.75, color=c, fontweight="bold")
    ax.set_ylabel("$\\hat{q}_{\\mathrm{ori}}$ (deg)", fontsize=10)
    ax.set_xlim(t0, t1)
    ax.set_ylim(0, 95)
    ax.tick_params(axis="both", labelsize=9)
    ax.set_title("Conformal uncertainty management\n(traj 13, seed 2)",
                 fontsize=9, fontweight="bold", pad=8)

    # Bottom: active Kp_ori
    ax = axes[1]
    ax.plot(t, log["kp_ori_active"], color="purple", linewidth=1.8, zorder=3,
            label="Active orientation gain $K_{p,\\mathrm{ori}}$")
    ax.axhline(5.0, color="black", ls=":", alpha=0.5, linewidth=1.2,
               label="$K_{p,\\mathrm{max}}=5$")
    ax.axhline(1.0, color="black", ls=":", alpha=0.5, linewidth=1.2,
               label="$K_{p,\\mathrm{min}}=1$")
    ax.set_ylabel("$K_{p,\\mathrm{ori}}$", fontsize=10)
    ax.set_ylim(0, 7)
    ax.set_xlim(t0, t1)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)

    kp = np.asarray(log["kp_ori_active"])
    gain_transition_idx = int(np.argmax(kp > 4.9))
    if 0 < gain_transition_idx < len(t):
        t_trans = t[gain_transition_idx]
        ax.annotate("$\\hat{q}_{\\mathrm{ori}}$ drops\nbelow 20°",
                    xy=(t_trans, kp[gain_transition_idx]),
                    xytext=(t_trans + 0.35, 2.2),
                    fontsize=8, color="purple", ha="left",
                    arrowprops=dict(arrowstyle="->", color="purple",
                                    alpha=0.7, lw=1.3))

    # Reserve the gaps FIRST, then attach a legend under each panel.
    # hspace -> room under the top panel; bottom -> room under the lower one.
    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.20, right=0.97, top=0.90,
                        bottom=0.26, hspace=0.47)
    legend_below(axes[0], ncol=2, yoff=-0.06, fontsize=8)
    legend_below(axes[1], ncol=2, yoff=-0.32, fontsize=8)

    plt.savefig(f"{OUT_DIR}/figure10_uncertainty_management.pdf")
    plt.savefig(f"{OUT_DIR}/figure10_uncertainty_management.png")
    plt.close()
    print("  -> figure10_uncertainty_management.{pdf,png}")

# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-traj", action="store_true",
                          help="Skip trajectory simulation (faster)")
    args = parser.parse_args()

    print("=" * 78)
    print("MASTER FIGURE & TABLE GENERATOR")
    print("=" * 78)
    print(f"Output directory: {OUT_DIR}/")

    # Load all data
    print("\nLoading results...")
    phase1 = load_results("path2_winning_v2_results.json")
    conf = load_results("path2_conformal_v3_results.json")
    with open("conformal_calibration_v2_repeated_subsample.json") as f:
        calib = json.load(f)
    print(f"  Phase 1 baseline: {len(phase1['omega'])} runs")
    print(f"  Conformal: {len(conf['omega'])} runs")
    print(f"  Calibration: alpha={calib['alpha']}")

    # Statistical figures (always)
    generate_statistical_figures(phase1, conf, calib)
    generate_tables(phase1, conf, calib)

    # Trajectory figures (require simulation, can skip)
    if args.skip_traj:
        print("\nSkipping trajectory figures (--skip-traj specified)")
    else:
        generate_trajectory_figures(calib)

    # Summary
    print()
    print("=" * 78)
    print("ALL FIGURES AND TABLES GENERATED")
    print("=" * 78)
    print(f"Files in {OUT_DIR}/:")
    for f in sorted(os.listdir(OUT_DIR)):
        size_kb = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
        print(f"  {f:50s}  {size_kb:>7.1f} KB")


if __name__ == "__main__":
    main()
