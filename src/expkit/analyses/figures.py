"""Static figures for the README, drawn only from results/*.json (python -m expkit.analyses.figures)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from expkit.analyses.common import FIGURES, read_json

BLUE, ORANGE, AQUA, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#e4e3df"


def _style(ax: Any) -> None:
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig: Any, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    p = FIGURES / name
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def aa_fpr() -> None:
    a = read_json("aa_tests")["methods"]
    keys = list(a)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    y = np.arange(len(keys))
    f = np.array([a[k]["fpr"] for k in keys])
    lo = np.array([a[k]["fpr_ci95"][0] for k in keys])
    hi = np.array([a[k]["fpr_ci95"][1] for k in keys])
    ax.axvspan(0.04, 0.06, color=GRID, alpha=0.6, label="acceptance band 4–6%")
    ax.errorbar(f, y, xerr=[f - lo, hi - f], fmt="o", color=BLUE, ms=6, capsize=3, label="FPR ± 95% CI")
    ax.axvline(0.05, color=INK2, lw=1, ls=":")
    ax.set_yticks(y, keys)
    ax.set_xlabel("false-positive rate at α = 0.05")
    ax.set_title(
        f"A/A on {read_json('aa_tests')['n_units']:,} real Criteo control users, {read_json('aa_tests')['n_reps']:,} re-randomizations",  # noqa: E501
        fontsize=10,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    _save(fig, "aa_fpr.png")


def power() -> None:
    p = read_json("power_curves")["curves"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (m, c) in zip(axes, p.items(), strict=True):
        x = [pt["relative_lift"] * 100 for pt in c["points"]]
        for k, col, lab in (
            ("raw", INK2, "Welch (raw)"),
            ("cuped", ORANGE, "CUPED (f0–f11)"),
            ("cupac", BLUE, "CUPAC"),
        ):
            ax.plot(
                x,
                [pt[f"power_{k}"] for pt in c["points"]],
                "o-",
                color=col,
                lw=2,
                ms=4,
                label=f"{lab} simulated",
            )
            ax.plot(x, [pt[f"analytic_{k}"] for pt in c["points"]], ":", color=col, lw=1.2)
        ax.axhline(0.8, color=INK2, lw=0.8, ls="--")
        ax.set_title(f"{m}: injected relative lift (dotted = analytic)", fontsize=9, loc="left")
        ax.set_xlabel("relative lift (%)")
        ax.set_ylabel("power")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=7)
    _save(fig, "power_curves.png")


def peeking() -> None:
    a = read_json("asos_portfolio")
    f = a["null_peeking_simulation"]["fpr"]
    s = a["real_peeking"]["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    names = ["fixed_horizon", "naive", "obf", "msprt"]
    labels = ["fixed horizon\n(look once)", "naive daily\npeeking", "O'Brien-\nFleming", "mSPRT"]
    vals = [f[k]["fpr"] for k in names]
    axes[0].bar(labels, vals, color=[INK2, ORANGE, BLUE, AQUA], width=0.6)
    axes[0].axhline(0.05, color=INK2, lw=1, ls=":")
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=8)
    axes[0].set_title("Null FPR on real ASOS look schedules (simulation)", fontsize=9, loc="left")
    _style(axes[0])
    meth = ["naive", "obf", "msprt"]
    unsup = [s[k]["declared_but_final_does_not_support"] for k in meth]
    sup = [s[k]["declared_significant"] - s[k]["declared_but_final_does_not_support"] for k in meth]
    lab2 = ["naive", "OBF", "mSPRT"]
    axes[1].bar(lab2, sup, color=BLUE, width=0.6, label="agrees with final analysis")
    axes[1].bar(lab2, unsup, bottom=sup, color=ORANGE, width=0.6, label="final analysis does not support")
    axes[1].set_title(f"Real ASOS trajectories (n = {s['n_trajectories']}): calls made", fontsize=9, loc="left")
    axes[1].legend(frameon=False, fontsize=7)
    _style(axes[1])
    _save(fig, "peeking.png")


def winners_curse() -> None:
    e = read_json("asos_portfolio")["eb"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for (m, d), col in zip(e.items(), [BLUE, ORANGE, AQUA, INK2], strict=True):
        rows = d.get("winners", {}).get("rows", [])
        ax.scatter(
            [r["observed"] * 100 for r in rows],
            [r["post_npmle"] * 100 for r in rows],
            color=col,
            s=18,
            label=m,
        )
    lim = ax.get_xlim()
    ax.plot(lim, lim, color=INK2, lw=0.8, ls=":", label="no shrinkage")
    ax.set_xlabel("observed lift of significant winners (%)")
    ax.set_ylabel("EB posterior mean (NPMLE prior, %)")
    ax.set_title("Winner's curse on the ASOS portfolio", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7)
    _style(ax)
    _save(fig, "winners_curse.png")


def heavy() -> None:
    h = read_json("heavy_tails")["full_sample"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    xs = [r["variance_reduction"] * 100 for r in h]
    ys = [r["relative_bias"] * 100 for r in h]
    ax.plot(xs, ys, "o-", color=BLUE, lw=2)
    for r, x, y in zip(h, xs, ys, strict=True):
        ax.annotate(r["cap"], (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7, color=INK2)
    ax.set_xlabel("variance reduction (%)")
    ax.set_ylabel("bias vs uncapped ATE (%)")
    ax.set_title("Capping Hillstrom spend: variance vs bias", fontsize=10, loc="left")
    _style(ax)
    _save(fig, "capping.png")


def main() -> None:
    for fn in (aa_fpr, power, peeking, winners_curse, heavy):
        try:
            fn()
        except FileNotFoundError as exc:
            print(f"skip {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
