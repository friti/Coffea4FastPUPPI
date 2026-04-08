import awkward as ak
import numpy as np

from utils import utils
from cut_config import ETA_RANGES, cut_range, cut_equal, apply_evt_mask

# ============================================================
# USER CONFIG (edit here and the bottom of the file)
# ============================================================

MODE = "pfpuppi"

ETA_REGION = "barrel"
PT_MIN = 1.0
DR_CUT = 0.1

GEN_PDGID = 130
RECO_PDGID = None

# Cone study
DO_CONE_STUDY = True
CONE_DRS = (0.1, 0.2, 0.3)

# Gen cuts
DO_GEN_STATUS1 = True

GEN_PTMIN_NEU = 1.0
GEN_PTMIN_CHG = 2.0

# ============================================================
# Geometry helpers
# ============================================================

def _delta_phi(a, b):
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi

def _delta_r(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2) ** 2 + _delta_phi(phi1, phi2) ** 2)

def _eta(x, useCalo):
    eta = ak.where(useCalo, x.caloeta, x.eta)
    return eta

def _phi(x, useCalo):
    phi = ak.where(useCalo, x.calophi, x.phi)
    return phi

# ============================================================
# Generic helpers
# ============================================================

def _split_chg_neu(coll):
    """
    Returns (charged, neutral) if coll.charge exists, else (None, None).
    """
    if coll is None or (not hasattr(coll, "charge")):
        return None, None
    m = (coll.charge != 0)
    return coll[m], coll[~m]

def _tag(dr):
    return f"dr{int(round(dr * 10)):02d}"

def _cone_counts_and_sums(
    centers,
    others,
    dr,
    *,
    exclude_self=False,
):
    """
    centers: [evt][Nc]
    others : [evt][No]
    Returns:
      n_in_cone [evt][Nc]
      sumpt     [evt][Nc]
    """
    #if centers is None or others is None:
    #    return None, None
    #if not hasattr(centers, "pt") or not hasattr(others, "pt"):
    #    return None, None

    pairs = ak.cartesian({"c": centers, "o": others}, axis=1, nested=True)
    c, o = pairs.c, pairs.o

    # use caloeta/calophi for neutrals only
    centers_calo = c.charge == 0 
    others_calo  = o.charge  == 0

    drs = _delta_r(
        _eta(c, centers_calo), _phi(c, centers_calo),
        _eta(o, others_calo),  _phi(o, others_calo),
    )
    mask = (drs < dr)

    if exclude_self:
        ic = ak.local_index(centers)
        io = ak.local_index(others)
        same = ak.cartesian({"ic": ic, "io": io}, axis=1, nested=True)
        mask = mask & (same.ic != same.io)

    #print(mask)
    n = ak.sum(mask, axis=2)
    s = ak.sum(ak.where(mask, o.pt, 0.0), axis=2)
    
    return n, s

def _closest_pt(ref, others, dr):
    """
    For each ref object, pt of closest 'others' within dr (else 0).
    Shapes:
      ref    [evt][Nref]
      others [evt][Nother]
    Returns:
      [evt][Nref]
    """
    if ref is None or others is None:
        return None
    if not hasattr(ref, "pt") or not hasattr(others, "pt"):
        return None

    # use caloeta/calophi for neutrals only
    ref_calo = ref.charge == 0
    others_calo  = others.charge  == 0

    reta = _eta(ref, ref_calo)
    rphi = _phi(ref, ref_calo)
    oeta = _eta(others, others_calo)
    ophi = _phi(others, others_calo)

    dphi = (rphi[:, :, None] - ophi[:, None, :] + np.pi) % (2 * np.pi) - np.pi
    deta = reta[:, :, None] - oeta[:, None, :]
    dr2 = dphi * dphi + deta * deta

    dr2 = ak.where(dr2 < dr * dr, dr2, np.inf)
    idx = ak.argmin(dr2, axis=2)
    ok = (ak.min(dr2, axis=2) < np.inf)
    
    return ak.where(ok, others[idx].pt, 0.0)

# ============================================================
# Gen selection
# ============================================================

def _apply_gen_cuts(gen):
    """
    Apply gen cuts safely with Awkward:
      - status == 1 (if exists)
      - pt thresholds split by charge (if exists)
    """
    if gen is None or (not hasattr(gen, "pt")):
        return gen

    m = ak.ones_like(gen.pt, dtype=np.bool_)

    if DO_GEN_STATUS1 and hasattr(gen, "status"):
        m = m & (gen.status == 1)

    if hasattr(gen, "charge"):
        is_chg = (gen.charge != 0)
        m = m & ak.where(is_chg, gen.pt > GEN_PTMIN_CHG, gen.pt > GEN_PTMIN_NEU)
    else:
        m = m & (gen.pt > GEN_PTMIN_NEU)

    return gen[m]

# ============================================================
# Cone metrics
# ============================================================

def _add_neutral_cone_metrics(out, reco_key, gen):
    """
    Decorate {reco_key}_neu with:
      n_gen_all_inCone_{drXX}, n_gen_chg_inCone_{drXX}, n_gen_neu_inCone_{drXX}
      ratio_a_{drXX}, ratio_b_{drXX}, ratio_c_{drXX}

    Uses caloeta/calophi for GEN NEUTRALS only (when available).
    """
    reco = out.get(reco_key)

    reco_chg, reco_neu = _split_chg_neu(reco)
    
    gen_chg, gen_neu = _split_chg_neu(gen)

    pt0 = reco_neu.pt
    eps = 1e-12

    for dr in CONE_DRS:
        tag = _tag(dr)

        # ----- reco denominators -----
        _, s_reco_chg = _cone_counts_and_sums(reco_neu, reco_chg, dr, exclude_self=False)
        _, s_reco_neu = _cone_counts_and_sums(reco_neu, reco_neu, dr, exclude_self=True)
        _, s_reco_all = _cone_counts_and_sums(reco_neu, reco,     dr, exclude_self=True)

        da = ak.where(s_reco_chg + pt0 > 0, s_reco_chg + pt0, eps)
        db = ak.where(s_reco_neu + pt0 > 0, s_reco_neu + pt0, eps)
        dc = ak.where(s_reco_all + pt0 > 0, s_reco_all + pt0, eps) 

        # ----- gen numerators -----
        n_all, s_all = _cone_counts_and_sums(reco_neu, gen, dr, exclude_self=False)

        if gen_chg is not None and gen_neu is not None:
            n_chg, s_chg = _cone_counts_and_sums(reco_neu, gen_chg, dr, exclude_self=False)
            n_neu, s_neu = _cone_counts_and_sums(reco_neu, gen_neu, dr, exclude_self=False)
        else:
            n_chg = ak.zeros_like(n_all)
            s_chg = ak.zeros_like(s_all)
            n_neu = ak.zeros_like(n_all)
            s_neu = ak.zeros_like(s_all)
        
        if gen_neu is not None:
            pt_closest = _closest_pt(reco_neu, gen_neu, dr)
        else:
            pt_closest = ak.zeros_like(pt0) # fill 0 if there's no closest gen neutral

        # ----- decorate -----
        reco_neu = ak.with_field(reco_neu, n_all, f"n_gen_all_inCone_{tag}")
        reco_neu = ak.with_field(reco_neu, n_chg, f"n_gen_chg_inCone_{tag}")
        reco_neu = ak.with_field(reco_neu, n_neu, f"n_gen_neu_inCone_{tag}")

        reco_neu = ak.with_field(reco_neu, (s_chg + pt_closest) / da, f"ratio_a_{tag}")
        reco_neu = ak.with_field(reco_neu, s_neu / db,               f"ratio_b_{tag}")
        reco_neu = ak.with_field(reco_neu, s_all / dc,               f"ratio_c_{tag}")

    out[f"{reco_key}_neu"] = reco_neu
    out[f"{reco_key}_chg"] = reco_chg
    return out

# ============================================================
# Matching
# ============================================================

def _decorate_isGenMatched(gen, reco):
    """
    Decorate reco with isGenMatched and matchedGenIdx.
    """
    m, idx = utils.match_reco_to_gen_indices(gen, reco, dr_cut=DR_CUT)
    best = utils.keep_highest_pt_reco_per_gen(reco, m, idx, pt_field="pt")
    m = m & best

    reco = ak.with_field(reco, m, "isGenMatched")
    reco = ak.with_field(reco, idx, "matchedGenIdx")
    return reco

def _split_matched(out, key):
    """
    Requires {key}_neu and {key}_chg already exist and have isGenMatched.
    Produces matched/nonMatched collections for both neutral and charged.
    """
    neu = out.get(f"{key}_neu")
    chg = out.get(f"{key}_chg")
    if neu is None or chg is None:
        return out
    if not hasattr(neu, "isGenMatched") or not hasattr(chg, "isGenMatched"):
        return out

    out[f"matched_{key}_neu"] = neu[neu.isGenMatched]
    out[f"nonMatched_{key}_neu"] = neu[~neu.isGenMatched]

    out[f"matched_{key}_chg"] = chg[chg.isGenMatched]
    out[f"nonMatched_{key}_chg"] = chg[~chg.isGenMatched]
    return out

# ============================================================
# Object building
# ============================================================

def build_objects(events):
    return {
        "pf": events.L1PFCands,
        "puppi": events.L1PuppiCands,
        "gen": events.GenCands,
    }

# ============================================================
# Cuts
# ============================================================

def cut_base(e, o):
    return o

def cut_pt(e, o):
    out = o
    for k in list(out.keys()):
        out = cut_range(e, out, k, "pt", vmin=PT_MIN)
    return out

def cut_eta(e, o):
    out = o
    r = ETA_RANGES[ETA_REGION]
    for k in list(out.keys()):
        out = cut_range(e, out, k, "eta", vmin=r[0], vmax=r[1], doAbs=True)
    return out

def cut_gen_status(e, o):
    out = dict(o)
    out["gen"] = _apply_gen_cuts(out.get("gen"))
    return out

def cut_matching(e, o):
    out = dict(o)
    out["pf"] = _decorate_isGenMatched(out["gen"], out["pf"])
    out["puppi"] = _decorate_isGenMatched(out["gen"], out["puppi"])

    # out["matched_gen"], out["nonMatched_gen"] = utils.get_genMatched(out["gen"], out["pf"], typ="Gen", dr_cut=DR_CUT)
    # out["matched_pf"], out["nonMatched_pf"], out["matched_pfTrue"] = utils.get_genMatched(out["gen"], out["pf"], typ="Reco", dr_cut=DR_CUT)
    # out["matched_puppi"], out["nonMatched_puppi"], out["matched_puppiTrue"] = utils.get_genMatched(out["gen"], out["puppi"], typ="Reco", dr_cut=DR_CUT)

    return out

def cut_cones(e, o):
    if not DO_CONE_STUDY:
        return o
    out = dict(o)
    out = _add_neutral_cone_metrics(out, "pf", out["gen"])
    out = _add_neutral_cone_metrics(out, "puppi", out["gen"])
    return out

def cut_split(e, o):
    out = dict(o)
    out = _split_matched(out, "pf")
    out = _split_matched(out, "puppi")
    return out

# ============================================================
# CUTFLOW
# ============================================================

CUTFLOW = [
    ("cut0_None", [cut_base]),
    ("cut1_pt", [cut_pt]),
    ("cut2_eta", [cut_eta]),
    (f"cut3_genStatusPt_ch{GEN_PTMIN_CHG}_ne{GEN_PTMIN_NEU}", [cut_gen_status]),
    ("cut4_matching", [cut_matching]),
    ("cut5_cones", [cut_cones]),
    ("cut6_splitMatched", [cut_split]),
]
