from asyncio import events

import awkward as ak
import numpy as np

from cut_config.cut_config_tkele import add_best_lead_sub
from cut_config import ETA_RANGES, cut_range, apply_evt_mask

# ============================================================
# USER CONFIG (only edit here & the bottom part of cutflow)
# ============================================================

MODE = "tkele_bkg"
ETA_REGION = "barrel"
PT_MIN = 4.0

PAIR_DVZ_MAX = 1.0
PAIR_MAX_RELPFISO = 1     # None to disable
PAIR_MAX_RELPUISO = 0.5   # None to disable

BESTPAIR_SCORE = "min_dvz"  # "pt" / "min_dvz" / "min_iso_pf"
BESTPAIR_PTMIN = 10.0       # None to disable

# isolation studies
DR_MAX_ISO = 0.3
SELF_VETO_DEFAULT = 0.02
SELF_VETO_LOOSER  = 0.05
OTHER_ELE_VETO_DR = 0.02
DZ_MAX_CHARGED    = 0.5

# ============================================================
# Helpers
# ============================================================

def _delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2 * np.pi) - np.pi

def _get_z(obj):
    return getattr(obj, "vz")

def add_rel_iso(coll):
    if coll is None or (not hasattr(coll, "pt")):
        return coll

    pt = coll.pt
    safe_pt = ak.where(pt > 0, pt, 1.0)

    if hasattr(coll, "pfIso"):
        coll = ak.with_field(coll, coll.pfIso / safe_pt, "relPfIso")
    if hasattr(coll, "puppiIso"):
        coll = ak.with_field(coll, coll.puppiIso / safe_pt, "relPuppiIso")

    return coll


def add_custom_cand_iso(events, ele, *,
                        cand_key,
                        dr_max=0.3,
                        # self-veto behavior
                        self_veto_dr=0.02,
                        self_veto_all=False,      # False => charged-only self-veto (original)
                        # partner-veto behavior
                        ele_other=None,
                        dr_partner_veto=0.02,
                        # dz behavior
                        dz_max=None,              # None disables charged dz cut
                        iso_name="customIso",
                        pt_field="pt",
                        use_ptcorr=True):
    """
    Generic candidate isolation:
      - sums cand pt for dR < dr_max
      - self-veto:
          * if self_veto_all: veto ALL candidates with dR < self_veto_dr
          * else: veto only CHARGED candidates with dR < self_veto_dr
      - optional partner veto: require dR(other,cand) > dr_partner_veto
      - optional charged dz cut: abs(z(cand)-z(ele)) < dz_max for charged cands
      - divides by ptCorr if present (and use_ptcorr) else pt
    """
    if ele is None:
        return None
    if not hasattr(events, cand_key):
        return ele
    if not (hasattr(ele, "eta") and hasattr(ele, "phi")):
        return ele

    denom = ele.ptCorr if (use_ptcorr and hasattr(ele, "ptCorr")) else (ele.pt if hasattr(ele, "pt") else None)
    if denom is None:
        return ele

    # mask candidates to same events as ele
    evt_mask = ak.num(ele, axis=1) >= 0
    cands = getattr(events, cand_key)[evt_mask]

    # [evt][ele][cand]
    ele_cand = ak.cartesian({"ele": ele, "cand": cands}, axis=1, nested=True)

    mask_neutral_ele = ele_cand.ele.charge == 0
    mask_neutral_pf = ele_cand.cand.charge == 0

    d_eta = ele_cand.ele.eta - ele_cand.cand.eta
    d_phi = _delta_phi(ele_cand.ele.phi, ele_cand.cand.phi)
    d_r = np.sqrt(d_eta * d_eta + d_phi * d_phi)

    keep = (d_r < dr_max)

    # self-veto
    if self_veto_all:
        keep = keep & ~(d_r < self_veto_dr)
    else:
        if hasattr(ele_cand.cand, "charge"):
            is_charged = (np.abs(ele_cand.cand.charge) > 0.5)
            keep = keep & ~((d_r < self_veto_dr) & is_charged)

    # partner-electron veto
    if ele_other is not None:
        eleO = ele_other[evt_mask]
        eleO_cand = ak.cartesian({"eleO": eleO, "cand": cands}, axis=1, nested=True)
        d_eta_O = eleO_cand.eleO.eta - eleO_cand.cand.eta
        d_phi_O = _delta_phi(eleO_cand.eleO.phi, eleO_cand.cand.phi)
        d_r_O = np.sqrt(d_eta_O * d_eta_O + d_phi_O * d_phi_O)
        keep = keep & (d_r_O > dr_partner_veto)

    # charged dz cut (z0 or vz)
    if (dz_max is not None) and hasattr(ele_cand.cand, "charge"):
        ele_z  = _get_z(ele_cand.ele)
        cand_z = _get_z(ele_cand.cand)
        if (ele_z is not None) and (cand_z is not None):
            is_charged = (np.abs(ele_cand.cand.charge) > 0.5)
            dz_ok = (np.abs(cand_z - ele_z) < dz_max)
            keep = keep & ((~is_charged) | dz_ok)

    cand_pt = getattr(ele_cand.cand, pt_field)
    pt_sum = ak.sum(ak.where(keep, cand_pt, 0.0), axis=2)

    denom_safe = ak.where(denom > 0, denom, 1.0)
    rel_iso = pt_sum / denom_safe

    return ak.with_field(ele, rel_iso, iso_name)


def add_all_custom_pfiso_versions(events, ele, dr_max=0.3):
    """
    Adds PF iso versions that do NOT require a partner electron:
      [1] customPfIso
      [2] customPfIso_looserSelfVeto
      [4] customPfIso_dz
    """
    if ele is None:
        return None
    if not hasattr(events, "L1PFCands"):
        return ele

    ele = add_custom_cand_iso(
        events, ele,
        cand_key="L1PFCands",
        dr_max=dr_max,
        self_veto_dr=SELF_VETO_DEFAULT,
        self_veto_all=False,
        dz_max=None,
        ele_other=None,
        iso_name="customPfIso",
    )

    ele = add_custom_cand_iso(
        events, ele,
        cand_key="L1PFCands",
        dr_max=dr_max,
        self_veto_dr=SELF_VETO_LOOSER,
        self_veto_all=True,    # include neutrals in self-veto
        dz_max=None,
        ele_other=None,
        iso_name="customPfIso_looserSelfVeto",
    )

    ele = add_custom_cand_iso(
        events, ele,
        cand_key="L1PFCands",
        dr_max=dr_max,
        self_veto_dr=SELF_VETO_LOOSER,
        self_veto_all=True,
        dz_max=DZ_MAX_CHARGED,
        ele_other=None,
        iso_name="customPfIso_dz",
    )

    return ele


def add_custom_puppi_iso(events, ele, dr_max=0.3):
    return add_custom_cand_iso(
        events, ele,
        cand_key="L1PuppiCands",
        dr_max=dr_max,
        self_veto_dr=SELF_VETO_DEFAULT,
        self_veto_all=False,
        dz_max=None,
        ele_other=None,
        iso_name="customPuppiIso",
    )


def add_pfiso_otherEleVeto_to_pair_legs(events, pairs, dr_max=0.3):
    """
    Computes [3] customPfIso_otherEleVeto on pair legs using the other leg as partner.
    Also adds pair-level max_customPfIso_otherEleVeto.
    """
    if pairs is None or (not hasattr(pairs, "l1")) or (not hasattr(pairs, "l2")):
        return pairs
    if not hasattr(events, "L1PFCands"):
        return pairs

    l1 = pairs.l1
    l2 = pairs.l2

    l1 = add_custom_cand_iso(
        events, l1,
        cand_key="L1PFCands",
        dr_max=dr_max,
        self_veto_dr=SELF_VETO_DEFAULT,
        self_veto_all=False,
        ele_other=l2,
        dr_partner_veto=OTHER_ELE_VETO_DR,
        dz_max=None,
        iso_name="customPfIso_otherEleVeto",
    )
    l2 = add_custom_cand_iso(
        events, l2,
        cand_key="L1PFCands",
        dr_max=dr_max,
        self_veto_dr=SELF_VETO_DEFAULT,
        self_veto_all=False,
        ele_other=l1,
        dr_partner_veto=OTHER_ELE_VETO_DR,
        dz_max=None,
        iso_name="customPfIso_otherEleVeto",
    )

    pairs = ak.with_field(pairs, l1, "l1")
    pairs = ak.with_field(pairs, l2, "l2")

    pairs = ak.with_field(
        pairs,
        np.maximum(pairs.l1.customPfIso_otherEleVeto, pairs.l2.customPfIso_otherEleVeto),
        "max_customPfIso_otherEleVeto"
    )
    return pairs

# get around weird awkward issue boh
def _as_jagged01(per_event_option):
    return ak.fill_none(ak.singletons(per_event_option), [])


def pick_best_pair(pair_coll, score="min_dvz"):
    if pair_coll is None:
        return None
    if score == "pt":
        idx = ak.argmax(pair_coll.pt, axis=1, keepdims=True)
    elif score == "min_dvz":
        idx = ak.argmin(pair_coll.delta_vz, axis=1, keepdims=True)
    elif score == "min_iso_pf":
        idx = ak.argmin(pair_coll.max_relPfIso, axis=1, keepdims=True)
    else:
        raise ValueError(score)
    return ak.firsts(pair_coll[idx])


def cut_pick_best_pair(events, obj, pair_key="tkelePair", out_key="best_tkelePair", score="min_dvz"):
    out = dict(obj)
    if pair_key not in out:
        return out
    best_opt = pick_best_pair(out[pair_key], score=score)
    out[out_key] = _as_jagged01(best_opt)
    return out


# ============================================================
# OBJECT BUILDING
# ============================================================

def build_objects(events):
    obj = {}
    if not hasattr(events, "TkEleL2"):
        return obj

    tkele = events.TkEleL2

    if hasattr(tkele, "ptCorr"): # piero regression
        tkele = ak.with_field(tkele, tkele.ptCorr, "pt")

    tkele = add_rel_iso(tkele)

    tkele = add_all_custom_pfiso_versions(events, tkele, dr_max=DR_MAX_ISO)

    #tkele = add_custom_puppi_iso(events, tkele, dr_max=DR_MAX_ISO)
    tkele = add_all_custom_puppi_iso_versions(events, tkele, dr_max=DR_MAX_ISO)

    obj["tkele"] = tkele

    if hasattr(events, "L1PFCands"):
        obj["L1PFCands"] = events.L1PFCands

    return obj


# ============================================================
# Pair building helpers
# ============================================================

def build_pairs_from_electrons(ele):
    pairs = ak.combinations(ele, 2, fields=["l1", "l2"])
    l1, l2 = pairs.l1, pairs.l2

    pt1, eta1, phi1 = l1.pt, l1.eta, l1.phi
    pt2, eta2, phi2 = l2.pt, l2.eta, l2.phi

    d_eta = eta1 - eta2
    d_phi = _delta_phi(phi1, phi2)

    m2 = 2.0 * pt1 * pt2 * (np.cosh(d_eta) - np.cos(d_phi))
    m2 = ak.where(m2 < 0, 0, m2)
    mass = np.sqrt(m2)

    px = pt1 * np.cos(phi1) + pt2 * np.cos(phi2)
    py = pt1 * np.sin(phi1) + pt2 * np.sin(phi2)
    pt = np.sqrt(px * px + py * py)
    phi = np.arctan2(py, px)

    pz = pt1 * np.sinh(eta1) + pt2 * np.sinh(eta2)
    eta = np.arcsinh(ak.where(pt == 0, 0, pz / pt))

    if hasattr(l1, "vz") and hasattr(l2, "vz"):
        vz = 0.5 * (l1.vz + l2.vz)
        d_vz = l1.vz - l2.vz
    else:
        vz = ak.zeros_like(pt)
        d_vz = ak.zeros_like(pt)

    d_pt = pt1 - pt2

    pairs = ak.with_field(pairs, pt, "pt")
    pairs = ak.with_field(pairs, eta, "eta")
    pairs = ak.with_field(pairs, phi, "phi")
    pairs = ak.with_field(pairs, vz, "vz")
    pairs = ak.with_field(pairs, mass, "mass")

    pairs = ak.with_field(pairs, abs(d_eta), "delta_eta")
    pairs = ak.with_field(pairs, abs(d_phi), "delta_phi")
    pairs = ak.with_field(pairs, abs(d_vz), "delta_vz")
    pairs = ak.with_field(pairs, abs(d_pt), "delta_pt")

    eps = 1e-12
    pt_safe  = ak.where(pt == 0, eps, pt)
    eta_safe = ak.where(eta == 0, eps, abs(eta))
    phi_safe = ak.where(phi == 0, eps, abs(phi))

    pairs = ak.with_field(pairs, abs(d_pt)  / pt_safe,  "delta_pt_over_pt")
    pairs = ak.with_field(pairs, abs(d_eta) / eta_safe, "delta_eta_over_eta")
    pairs = ak.with_field(pairs, abs(d_phi) / phi_safe, "delta_phi_over_phi")

    pt_over_mass = ak.where(mass > 0, pt / mass, 0.0)
    pairs = ak.with_field(pairs, pt_over_mass, "pt_over_mass")

    if hasattr(l1, "charge") and hasattr(l2, "charge"):
        pairs = ak.with_field(pairs, l1.charge * l2.charge, "charge_prod")

    pairs = ak.with_field(pairs, l1.eta * l2.eta, "eta_prod")

    if hasattr(l1, "relPfIso") and hasattr(l2, "relPfIso"):
        pairs = ak.with_field(pairs, np.maximum(l1.relPfIso, l2.relPfIso), "max_relPfIso")
    if hasattr(l1, "relPuppiIso") and hasattr(l2, "relPuppiIso"):
        pairs = ak.with_field(pairs, np.maximum(l1.relPuppiIso, l2.relPuppiIso), "max_relPuppiIso")

    d_r = np.sqrt(d_eta * d_eta + d_phi * d_phi)
    pairs = ak.with_field(pairs, d_r, "delta_r")

    if hasattr(l1, "customPfIso") and hasattr(l2, "customPfIso"):
        pairs = ak.with_field(pairs, np.maximum(l1.customPfIso, l2.customPfIso), "max_customPfIso")
    if hasattr(l1, "customPfIso_looserSelfVeto") and hasattr(l2, "customPfIso_looserSelfVeto"):
        pairs = ak.with_field(
            pairs,
            np.maximum(l1.customPfIso_looserSelfVeto, l2.customPfIso_looserSelfVeto),
            "max_customPfIso_looserSelfVeto"
        )
    if hasattr(l1, "customPfIso_dz") and hasattr(l2, "customPfIso_dz"):
        pairs = ak.with_field(pairs, np.maximum(l1.customPfIso_dz, l2.customPfIso_dz), "max_customPfIso_dz")

    if hasattr(l1, "customPuppiIso") and hasattr(l2, "customPuppiIso"):
        pairs = ak.with_field(pairs, np.maximum(l1.customPuppiIso, l2.customPuppiIso), "max_customPuppiIso")

    return pairs


def build_lead_sub_from_pairs(pair_coll):
    if pair_coll is None or (not hasattr(pair_coll, "l1")) or (not hasattr(pair_coll, "l2")):
        return None, None
    l1, l2 = pair_coll.l1, pair_coll.l2
    if not (hasattr(l1, "pt") and hasattr(l2, "pt")):
        return None, None
    lead_mask = (l1.pt >= l2.pt)
    lead = ak.where(lead_mask, l1, l2)
    sub  = ak.where(lead_mask, l2, l1)
    return lead, sub


def add_all_custom_puppi_iso_versions(events, ele, dr_max=0.3):

    if ele is None:
        return None
    if not hasattr(events, "L1PuppiCands"):
        return ele

    # [1] baseline
    ele = add_custom_cand_iso(
        events, ele,
        cand_key="L1PuppiCands",
        dr_max=dr_max,
        self_veto_dr=SELF_VETO_DEFAULT,
        self_veto_all=False,
        dz_max=None,
        ele_other=None,
        iso_name="customPuppiIso",
    )

    # [2] looser self-veto
    ele = add_custom_cand_iso(
        events, ele,
        cand_key="L1PuppiCands",
        dr_max=dr_max,
        self_veto_dr=SELF_VETO_LOOSER,
        self_veto_all=True,
        dz_max=None,
        ele_other=None,
        iso_name="customPuppiIso_looserSelfVeto",
    )

    # [3] dz cut
    ele = add_custom_cand_iso(
        events, ele,
        cand_key="L1PuppiCands",
        dr_max=dr_max,
        self_veto_dr=SELF_VETO_LOOSER,
        self_veto_all=True,
        dz_max=DZ_MAX_CHARGED,
        ele_other=None,
        iso_name="customPuppiIso_dz",
    )

    return ele

def add_puppiiso_otherEleVeto_to_pair_legs(events, pairs, dr_max=0.3):

    if pairs is None or (not hasattr(pairs, "l1")):
        return pairs
    if not hasattr(events, "L1PuppiCands"):
        return pairs

    l1 = pairs.l1
    l2 = pairs.l2

    l1 = add_custom_cand_iso(
        events, l1,
        cand_key="L1PuppiCands",
        dr_max=dr_max,
        self_veto_dr=SELF_VETO_LOOSER,
        self_veto_all=True,
        ele_other=l2,
        dr_partner_veto=OTHER_ELE_VETO_DR,
        dz_max=DZ_MAX_CHARGED,
        iso_name="customPuppiIso_otherEleVeto",
    )

    l2 = add_custom_cand_iso(
        events, l2,
        cand_key="L1PuppiCands",
        dr_max=dr_max,
        self_veto_dr=SELF_VETO_LOOSER,
        self_veto_all=True,
        ele_other=l1,
        dr_partner_veto=OTHER_ELE_VETO_DR,
        dz_max=DZ_MAX_CHARGED,
        iso_name="customPuppiIso_otherEleVeto",
    )

    pairs = ak.with_field(pairs, l1, "l1")
    pairs = ak.with_field(pairs, l2, "l2")

    pairs = ak.with_field(
        pairs,
        np.maximum(pairs.l1.customPuppiIso_otherEleVeto,
                   pairs.l2.customPuppiIso_otherEleVeto),
        "max_customPuppiIso_otherEleVeto"
    )

    return pairs

# ============================================================
# CUTS
# ============================================================

def cut_base(events, obj):
    return obj

def cut_eta(events, obj):
    eta_min, eta_max = ETA_RANGES[ETA_REGION]
    out = obj
    if "tkele" in out:
        out = cut_range(events, out, "tkele", "eta", vmin=eta_min, vmax=eta_max, doAbs=True)
        out["tkele"] = add_rel_iso(out["tkele"])
        out["tkele"] = add_all_custom_pfiso_versions(events, out["tkele"], dr_max=DR_MAX_ISO)
        #out["tkele"] = add_custom_puppi_iso(events, out["tkele"], dr_max=DR_MAX_ISO)
        out["tkele"] = add_all_custom_puppi_iso_versions(events, out["tkele"], dr_max=DR_MAX_ISO)
    return out

def cut_pt(events, obj):
    out = obj
    if "tkele" in out:
        out = cut_range(events, out, "tkele", "pt", vmin=PT_MIN, vmax=None, doAbs=False)
        out["tkele"] = add_rel_iso(out["tkele"])
        out["tkele"] = add_all_custom_pfiso_versions(events, out["tkele"], dr_max=DR_MAX_ISO)
        #out["tkele"] = add_custom_puppi_iso(events, out["tkele"], dr_max=DR_MAX_ISO)
        out["tkele"] = add_all_custom_puppi_iso_versions(events, out["tkele"], dr_max=DR_MAX_ISO)
    return out

def cut_build_pairs(events, obj):
    out = dict(obj)
    if "tkele" not in out:
        return out

    # build pairs from electrons that already have [1],[2],[4] PF iso variants + customPuppiIso
    pairs = build_pairs_from_electrons(out["tkele"])

    # add [3] (partner veto) on pair legs + pair-level max_customPfIso_otherEleVeto
    #pairs = add_pfiso_otherEleVeto_to_pair_legs(events, pairs, dr_max=DR_MAX_ISO)
    pairs = add_pfiso_otherEleVeto_to_pair_legs(events, pairs, dr_max=DR_MAX_ISO)
    #pairs = add_puppiiso_otherEleVeto_to_pair_legs(events, pairs, dr_max=DR_MAX_ISO)

    out["tkelePair"] = pairs

    # lead/sub inherit the decorated leg fields (including otherEleVeto)
    out["tkeleLead"], out["tkeleSub"] = build_lead_sub_from_pairs(out["tkelePair"])

    out["tkeleLead"] = add_rel_iso(out["tkeleLead"])
    out["tkeleSub"]  = add_rel_iso(out["tkeleSub"])

    return out


def cut_pair_os(obj, pair_key="tkelePair"):
    out = dict(obj)
    if pair_key not in out:
        return out
    p = out[pair_key]
    if p is None or (not hasattr(p, "charge_prod")):
        return out
    out[pair_key] = p[p.charge_prod < 0]
    return out


def cut_pair_dvz(obj, pair_key="tkelePair", dvz_max=1.0):
    out = dict(obj)
    if pair_key not in out:
        return out
    p = out[pair_key]
    if p is None or (not hasattr(p, "delta_vz")):
        return out
    out[pair_key] = p[p.delta_vz < dvz_max]
    return out


def cut_pair_max_reliso(obj, pair_key="tkelePair", max_relPfIso=None, max_relPuppiIso=None):
    out = dict(obj)
    if pair_key not in out:
        return out
    p = out[pair_key]
    if p is None:
        return out

    mask = ak.ones_like(p.pt, dtype=bool)
    if (max_relPfIso is not None) and hasattr(p, "max_relPfIso"):
        mask = mask & (p.max_relPfIso < max_relPfIso)
    if (max_relPuppiIso is not None) and hasattr(p, "max_relPuppiIso"):
        mask = mask & (p.max_relPuppiIso < max_relPuppiIso)

    out[pair_key] = p[mask]
    return out


def cut_veto_if_no_bestpair(events, obj, best_key="best_tkelePair"):
    if best_key not in obj:
        return obj
    bp = obj[best_key]
    if bp is None:
        return obj
    evt_mask = ak.num(bp, axis=1) > 0
    return apply_evt_mask(obj, evt_mask)


def cut_event_on_bestpair(events, obj, best_key="best_tkelePair", *, ptmin=None):
    out = dict(obj)
    if best_key not in out:
        return out

    bp = ak.firsts(out[best_key])  # [evt] option-record
    mask = ~ak.is_none(bp)

    if ptmin is not None and hasattr(bp, "pt"):
        mask = mask & (ak.fill_none(bp.pt, -1.0) >= ptmin)

    return apply_evt_mask(out, mask)


def cut_bestpair_leg_pt(events, obj,
                        lead_key="best_tkeleLead",
                        sub_key="best_tkeleSub",
                        lead_pt_min=5.0,
                        sub_pt_min=4.0):
    out = dict(obj)
    if (lead_key not in out) or (sub_key not in out):
        return out

    lead = out[lead_key]
    sub  = out[sub_key]
    if lead is None or sub is None:
        return out

    lead1 = ak.firsts(lead)
    sub1  = ak.firsts(sub)

    mask = (~ak.is_none(lead1)) & (~ak.is_none(sub1))

    if hasattr(lead1, "pt"):
        mask = mask & (ak.fill_none(lead1.pt, -1.0) > lead_pt_min)
    if hasattr(sub1, "pt"):
        mask = mask & (ak.fill_none(sub1.pt, -1.0) > sub_pt_min)

    return apply_evt_mask(out, mask)


def cut_bestpair_leg_idscore(events, obj,
                             lead_key="best_tkeleLead",
                             sub_key="best_tkeleSub",
                             id_min=-0.1):
    out = dict(obj)
    if (lead_key not in out) or (sub_key not in out):
        return out

    lead = out[lead_key]
    sub  = out[sub_key]
    if lead is None or sub is None:
        return out

    lead1 = ak.firsts(lead)
    sub1  = ak.firsts(sub)

    mask = (~ak.is_none(lead1)) & (~ak.is_none(sub1))

    if hasattr(lead1, "idScore"):
        mask = mask & (ak.fill_none(lead1.idScore, -999.0) > id_min)
    if hasattr(sub1, "idScore"):
        mask = mask & (ak.fill_none(sub1.idScore, -999.0) > id_min)

    return apply_evt_mask(out, mask)


# ============================================================
# CUTFLOW
# ============================================================

CUTFLOW = [
    ("cut0_None",              [cut_base]),
    (f"cut1_eta_{ETA_REGION}", [cut_eta]),
    (f"cut2_pt_{int(PT_MIN)}", [cut_pt]),

    ("cut3_buildPairs",        [cut_build_pairs,
                                lambda e, o: cut_pick_best_pair(e, o, score=BESTPAIR_SCORE),
                                lambda e, o: add_best_lead_sub(o, best_key="best_tkelePair")]),
]
