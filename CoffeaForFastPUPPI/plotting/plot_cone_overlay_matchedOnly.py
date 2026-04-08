#!/usr/bin/env python3
"""
plot_cone_overlay_ratios_matchedonly.py

For each dr tag (dr01, dr02, dr03, ...), overlay matched-only:
  - ratio_a_<tag>
  - ratio_b_<tag>
  - ratio_c_<tag>

Legend mapping:
  ratio_a -> charged gen around reco neutral
  ratio_b -> neutral gen around reco neutral
  ratio_c -> all gen around reco neutral

Looks for histograms in a stage dict with names:
  matched_pf_neu_ratio_a_dr01   (etc)
  matched_pf_neu_ratio_b_dr01
  matched_pf_neu_ratio_c_dr01

and similarly for matched_puppi_neu_...

Usage:
  python plot_cone_overlay_ratios_matchedonly.py coffea/FILE.coffea \
    --stage cut6_splitMatched --outdir overlays_ratios --which both --density
"""

import os
import argparse
import re
import warnings

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

LEGEND_LABELS = {
    "ratio_a": "charged gen around reco neutral",
    "ratio_b": "neutral gen around reco neutral",
    "ratio_c": "all gen around reco neutral",
}

ORDER = ["ratio_a", "ratio_b", "ratio_c"]
LINESTYLE = {"ratio_a": "-", "ratio_b": "--", "ratio_c": ":"}


def _ensure_dir(d):
    if d:
        os.makedirs(d, exist_ok=True)


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
    ax0 = h.axes[0]
    edges = np.asarray(ax0.edges, dtype=float)
    vals = np.asarray(h.values(flow=False), dtype=float)
    return edges, vals, ax0


def _density_normalize_1d(edges, vals):
    widths = np.diff(edges)
    area = np.sum(vals * widths)
    if area > 0:
        return vals / area
    return vals


def _step(ax, edges, vals, *, label, linestyle="-", linewidth=2.0):
    if len(vals) == 0:
        return
    y = np.r_[vals, vals[-1]]
    ax.step(edges, y, where="post", label=label, linestyle=linestyle, linewidth=linewidth)


def _find_tags(stage_hists, prefix):
    """
    From keys like:
      matched_pf_neu_ratio_a_dr01
    extract tags "dr01", etc (only for ratio_[abc]).
    """
    tags = set()
    pat = re.compile(rf"^{re.escape(prefix)}(ratio_[abc])_(dr\d+)$")
    for name in stage_hists.keys():
        m = pat.match(name)
        if m:
            tags.add(m.group(2))
    return sorted(tags)


def _get_ratio_hist(stage_hists, prefix, ratio_key, tag):
    name = f"{prefix}{ratio_key}_{tag}"
    return stage_hists.get(name, None)


def _plot_one_canvas(outpath, hists_by_ratio, *, density=False, logy=True, title=None, xlabel=None):
    """
    hists_by_ratio: dict ratio_key -> Hist
    Must all be 1D and same binning.
    """
    # drop missing/empty
    use = []
    for rk in ORDER:
        h = hists_by_ratio.get(rk)
        if h is None:
            continue
        if (not _is_1d(h)) or _hist_is_empty(h):
            continue
        use.append((rk, h))

    if len(use) == 0:
        return False

    # reference binning
    edges0, vals0, ax0 = _edges_vals_axis_1d(use[0][1])
    for rk, h in use[1:]:
        edges, _, _ = _edges_vals_axis_1d(h)
        if edges.shape != edges0.shape or not np.allclose(edges, edges0, rtol=0, atol=1e-12):
            # refuse to overlay if binning differs
            return False

    fig, ax = plt.subplots()

    for rk, h in use:
        edges, vals, _ = _edges_vals_axis_1d(h)
        if density:
            vals = _density_normalize_1d(edges, vals)

        _step(
            ax, edges, vals,
            label=LEGEND_LABELS.get(rk, rk),
            linestyle=LINESTYLE.get(rk, "-"),
            linewidth=2.2,
        )

    #hep.cms.text("Phase-2 Simulation", ax=ax)

    if xlabel is None:
        xlabel = getattr(ax0, "label", ax0.name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("A.U." if density else "Entries")
    if title:
        ax.set_title(title)

    ax.set_yscale("log" if logy else "linear")

    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    _ensure_dir(os.path.dirname(outpath))
    fig.savefig(outpath)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coffea_file", help="Input .coffea file")
    ap.add_argument("--stage", default="cut6_splitMatched", help="Stage name")
    ap.add_argument("--outdir", default="overlays_ratios_matched", help="Output directory")
    ap.add_argument("--which", choices=["pf", "puppi", "both"], default="both", help="Which reco collection")
    ap.add_argument("--density", action="store_true", help="Normalize each curve to unit area")
    ap.add_argument("--linear", action="store_true", help="Use linear y (default is logy for ratios)")
    args = ap.parse_args()

    res = util.load(args.coffea_file)
    stage_hists = (res.get("hists", {}) or {}).get(args.stage, {})
    if not stage_hists:
        raise SystemExit(f"No histograms found for stage='{args.stage}' in '{args.coffea_file}'")

    jobs = []
    if args.which in ("pf", "both"):
        jobs.append(("pf", "matched_pf_neu_"))
    if args.which in ("puppi", "both"):
        jobs.append(("puppi", "matched_puppi_neu_"))

    n_written = 0
    n_attempt = 0

    for coll_tag, prefix in jobs:
        tags = _find_tags(stage_hists, prefix)
        if not tags:
            print(f"[warn] No ratio histograms found for prefix='{prefix}' in stage '{args.stage}'")
            continue

        for tag in tags:
            n_attempt += 1
            hists = {
                "ratio_a": _get_ratio_hist(stage_hists, prefix, "ratio_a", tag),
                "ratio_b": _get_ratio_hist(stage_hists, prefix, "ratio_b", tag),
                "ratio_c": _get_ratio_hist(stage_hists, prefix, "ratio_c", tag),
            }

            outpath = os.path.join(args.outdir, coll_tag, f"{coll_tag}_ratios_{tag}.png")
            title = f"Matched neutrals: {tag}"
            ok = _plot_one_canvas(
                outpath,
                hists,
                density=args.density,
                logy=(not args.linear),
                title=title,
                xlabel=f"ratio (cone {tag})",
            )
            if ok:
                n_written += 1

    print(f"[plot_cone_overlay_ratios_matchedonly] wrote {n_written}/{n_attempt} plots -> {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
