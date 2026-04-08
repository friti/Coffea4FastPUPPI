#!/usr/bin/env python3
"""
plot_cone_overlay_matched.py

Overlay matched vs nonMatched cone-study histograms (multiplicities + ratios)
for PF and/or PUPPI, with:
  - multiplicity plots: linear y
  - ratio plots       : log y

It auto-discovers histograms in the chosen stage whose names start with:
  matched_pf_neu_ / nonMatched_pf_neu_
  matched_puppi_neu_ / nonMatched_puppi_neu_
(and overlays matched/nonMatched for each common suffix)

Usage:
  python plot_cone_overlay_matched.py coffea/TTpu0_barrel_genPtCut.coffea \
    --stage cut6_splitMatched --outdir overlays_pfpuppi --which both --density
"""

import os
import argparse
import warnings
import re

import numpy as np
import coffea.util as util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")

warnings.filterwarnings(
    "ignore",
    message=r".*All sumw are zero!.*",
    category=RuntimeWarning,
)


def _hist_is_empty(h):
    try:
        vals = np.asarray(h.values(flow=False), dtype=float)
        return (vals.size == 0) or (np.nansum(vals) == 0.0)
    except Exception:
        return True


def _is_1d(h):
    try:
        return len(h.axes) == 1
    except Exception:
        return False


def _edges_vals_axis_1d(h):
    """Only call this for 1D histograms."""
    ax0 = h.axes[0]
    edges = np.asarray(ax0.edges, dtype=float)
    vals = np.asarray(h.values(flow=False), dtype=float)
    return edges, vals, ax0


def _density_normalize_1d(edges, vals):
    widths = np.diff(edges)
    # vals must be 1D
    area = np.sum(vals * widths)
    if area > 0:
        return vals / area
    return vals


def _step(ax, edges, vals, *, label, linestyle="-", linewidth=2.0):
    # extend last bin to final edge
    if len(vals) == 0:
        return
    y = np.r_[vals, vals[-1]]
    ax.step(edges, y, where="post", label=label, linestyle=linestyle, linewidth=linewidth)


def _is_ratio(name_or_suffix: str) -> bool:
    return bool(re.search(r"(^|_)ratio_[abc](_|$)", name_or_suffix))


def _is_multiplicity(name_or_suffix: str) -> bool:
    return bool(re.search(r"(^|_)n_gen_(all|chg|neu)_inCone_", name_or_suffix))


def _ensure_dir(d):
    if d:
        os.makedirs(d, exist_ok=True)


def _collect_pairs(stage_hists: dict, matched_prefix: str, nonmatched_prefix: str):
    """
    Returns list of (suffix, h_matched, h_nonmatched) for common suffixes.
    """
    matched = {}
    nonmatched = {}

    for name, h in stage_hists.items():
        if name.startswith(matched_prefix):
            matched[name[len(matched_prefix):]] = h
        elif name.startswith(nonmatched_prefix):
            nonmatched[name[len(nonmatched_prefix):]] = h

    suffixes = sorted(set(matched.keys()) & set(nonmatched.keys()))
    return [(sfx, matched[sfx], nonmatched[sfx]) for sfx in suffixes]


def _plot_one_1d(outpath, h_m, h_nm, *, density=False, title=None, force_logy=None):
    # both empty -> skip
    if _hist_is_empty(h_m) and _hist_is_empty(h_nm):
        return False

    # must be 1D
    if not (_is_1d(h_m) and _is_1d(h_nm)):
        return False

    edges_m, vals_m, ax0 = _edges_vals_axis_1d(h_m)
    edges_n, vals_n, _ = _edges_vals_axis_1d(h_nm)

    # require same binning
    if edges_m.shape != edges_n.shape or not np.allclose(edges_m, edges_n, rtol=0, atol=1e-12):
        return False

    edges = edges_m

    if density:
        vals_m = _density_normalize_1d(edges, vals_m)
        vals_n = _density_normalize_1d(edges, vals_n)

    fig, ax = plt.subplots()

    _step(ax, edges, vals_m, label="matched", linestyle="-", linewidth=2.0)
    #_step(ax, edges, vals_n, label="nonMatched", linestyle="--", linewidth=2.0)

    hep.cms.text("Phase-2 Simulation", ax=ax)

    xlabel = getattr(ax0, "label", ax0.name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("A.U." if density else "Entries")

    if title:
        ax.set_title(title)

    if force_logy is not None:
        ax.set_yscale("log" if force_logy else "linear")

    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    _ensure_dir(os.path.dirname(outpath))
    fig.savefig(outpath)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coffea_file", help="Input .coffea file")
    ap.add_argument("--stage", default="cut6_splitMatched", help="Stage name (default: cut6_splitMatched)")
    ap.add_argument("--outdir", default="overlays_cone", help="Output directory")
    ap.add_argument("--which", choices=["pf", "puppi", "both"], default="both", help="Which reco collection")
    ap.add_argument("--density", action="store_true", help="Normalize each curve to unit area (1D only)")
    ap.add_argument(
        "--logy",
        action="store_true",
        help="Force logy for ALL 1D plots. If not set: ratios=log, multiplicities=linear.",
    )
    args = ap.parse_args()

    res = util.load(args.coffea_file)
    stage_hists = (res.get("hists", {}) or {}).get(args.stage, {})
    if not stage_hists:
        raise SystemExit(f"No histograms found for stage='{args.stage}' in '{args.coffea_file}'")

    todo = []
    if args.which in ("pf", "both"):
        todo.append(("pf", "matched_pf_neu_", "nonMatched_pf_neu_"))
        # If you also produce charged collections:
        # todo.append(("pf_chg", "matched_pf_chg_", "nonMatched_pf_chg_"))
    if args.which in ("puppi", "both"):
        todo.append(("puppi", "matched_puppi_neu_", "nonMatched_puppi_neu_"))
        # todo.append(("puppi_chg", "matched_puppi_chg_", "nonMatched_puppi_chg_"))

    n_written = 0
    n_seen = 0
    n_skipped_nd = 0
    n_skipped_empty = 0

    for tagname, pref_m, pref_nm in todo:
        pairs = _collect_pairs(stage_hists, pref_m, pref_nm)

        for suffix, h_m, h_nm in pairs:
            n_seen += 1

            # Skip non-1D (this avoids your broadcast error)
            if not (_is_1d(h_m) and _is_1d(h_nm)):
                n_skipped_nd += 1
                continue

            # Skip if both empty
            if _hist_is_empty(h_m) and _hist_is_empty(h_nm):
                n_skipped_empty += 1
                continue

            # y-scale rule
            if args.logy:
                force_logy = True
            else:
                if _is_ratio(suffix):
                    force_logy = False
                elif _is_multiplicity(suffix):
                    force_logy = False
                else:
                    force_logy = True if "ratio" in suffix else False

            outpath = os.path.join(args.outdir, tagname, f"{tagname}_{suffix}.png")
            title = suffix.replace("_", " ")

            ok = _plot_one_1d(
                outpath,
                h_m,
                h_nm,
                density=args.density,
                title=title,
                force_logy=force_logy,
            )
            if ok:
                n_written += 1

    print(
        f"[plot_cone_overlay_matched] stage='{args.stage}' wrote {n_written}/{n_seen} overlays -> {args.outdir}\n"
        f"  skipped non-1D: {n_skipped_nd}\n"
        f"  skipped empty : {n_skipped_empty}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
