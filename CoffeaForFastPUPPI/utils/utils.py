import numpy as np
import awkward as ak

def delta_phi(phi1, phi2):
    # Compute delta phi between two angles, handling the periodicity
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2 * np.pi) - np.pi

def deltaR(eta1, phi1, eta2, phi2):
    # Compute delta R between two objects given their eta and phi
    dphi = delta_phi(phi1, phi2)
    deta = eta1 - eta2
    return np.sqrt(deta**2 + dphi**2)

def cut_hgcIdPu(pf, wp=0.5):
    # Apply HGC ID PU, PI, EM cuts to PF candidates
    passPuId = pf.hgcIdPu < wp
    passPiId = passPuId & (pf.hgcIdPi > pf.hgcIdEm)
    passEmId = passPuId & (pf.hgcIdPi < pf.hgcIdEm)
    return pf[passPuId], pf[passPiId], pf[passEmId]

def get_genMatched(gen, reco, typ="Gen", dr_cut=0.1):
    """
    typ='Gen':  returns (matched_gen, nonMatched_gen)
      - matched_gen: gen objects that have >=1 reco within dr_cut
    typ='Reco': returns (matched_reco, nonMatched_reco, matchedGenForReco)
      - matched_reco: reco objects that have >=1 gen within dr_cut
      - matchedGenForReco: the closest gen object for each matched reco
    """
    if typ == "Gen":
        # dR shape: [events][gen][reco]
        pairs = ak.cartesian([gen, reco], axis=1, nested=True)
        gen_cands, reco_cands = ak.unzip(pairs)

        # caloeta, calophi for gen neutral
        useCalo = gen_cands.charge == 0 # use caloeta/calophi for neutrals only
        eta_gen = ak.where(useCalo, gen_cands.caloeta, gen_cands.eta)
        phi_gen = ak.where(useCalo, gen_cands.calophi, gen_cands.phi)

        dEta = np.abs(eta_gen - reco_cands.eta)
        dPhi = np.abs(phi_gen - reco_cands.phi)
        dPhi = ak.where(dPhi > np.pi, 2*np.pi - dPhi, dPhi)
        dR = np.sqrt(dEta**2 + dPhi**2)

        # closest reco per gen -> [events][gen]
        min_dR = ak.min(dR, axis=-1)
        min_dR = ak.fill_none(min_dR, 999.)
        matched_mask = (min_dR < dr_cut)

        matched = gen[matched_mask]
        nonMatched = gen[~matched_mask]
        return matched, nonMatched

    elif typ == "Reco":
        # dR shape: [events][reco][gen]
        pairs = ak.cartesian([reco, gen], axis=1, nested=True)
        reco_cands, gen_cands = ak.unzip(pairs)

        useCalo = gen_cands.charge == 0 # use caloeta/calophi for neutrals only
        eta_gen = ak.where(useCalo, gen_cands.caloeta, gen_cands.eta)
        phi_gen = ak.where(useCalo, gen_cands.calophi, gen_cands.phi)

        dEta = np.abs(reco_cands.eta - eta_gen)
        dPhi = np.abs(reco_cands.phi - phi_gen)
        dPhi = ak.where(dPhi > np.pi, 2*np.pi - dPhi, dPhi)
        dR = np.sqrt(dEta**2 + dPhi**2)

        # closest gen per reco -> [events][reco]
        min_dR = ak.min(dR, axis=-1)
        min_dR = ak.fill_none(min_dR, 999.)
        matched_mask = (min_dR < dr_cut)

        matched = reco[matched_mask]
        nonMatched = reco[~matched_mask]

        # argmin gen index per reco -> [events][reco]
        idx = ak.argmin(dR, axis=-1)
        matchedGen = gen[idx][matched_mask]

        return matched, nonMatched, matchedGen

    else:
        raise ValueError(f"Unknown typ='{typ}', expected 'Gen' or 'Reco'")


def match_reco_to_gen_indices(gen, reco, dr_cut=0.1):
    """
    For each reco object, find closest gen object and decide if matched.
    Returns:
      matched_mask: [events][reco] boolean
      matched_gen_idx: [events][reco] int (best gen index; -1 if not matched or invalid)
    SAFE against events with 0 gen objects and against any out-of-range indices.
    """
    # pairs: [events][reco][gen]
    pairs = ak.cartesian([reco, gen], axis=1, nested=True)
    reco_cands, gen_cands = ak.unzip(pairs)

    # use caloeta/calophi for neutrals only
    useCalo = gen_cands.charge == 0
    eta_gen = ak.where(useCalo, gen_cands.caloeta, gen_cands.eta)
    phi_gen = ak.where(useCalo, gen_cands.calophi, gen_cands.phi)

    deta = reco_cands.eta - eta_gen
    dphi = np.abs(reco_cands.phi - phi_gen)
    dphi = ak.where(dphi > np.pi, 2*np.pi - dphi, dphi)
    dr = np.sqrt(deta**2 + dphi**2)

    min_dr = ak.min(dr, axis=-1)
    min_dr = ak.fill_none(min_dr, 999.0)

    best_idx = ak.argmin(dr, axis=-1)      # [events][reco] (always defined)
    matched_mask = (min_dr < dr_cut)       # [events][reco]

    # ---- critical safety: clamp indices to [0, n_gen-1] per event ----
    n_gen = ak.num(gen, axis=1)            # [events]
    n_gen = n_gen[:, None]                 # broadcast to [events][reco]

    valid_idx = (best_idx >= 0) & (best_idx < n_gen)
    matched_mask = matched_mask & valid_idx

    matched_gen_idx = ak.where(matched_mask, best_idx, -1)

    return matched_mask, matched_gen_idx


def keep_highest_pt_reco_per_gen(reco, matched_mask, matched_gen_idx, pt_field="pt"):
    """
    Keep at most ONE reco per gen per event: the matched reco with the highest pT.

    Inputs
    ------
    reco : jagged array [events][reco]
    matched_mask : jagged bool [events][reco]
        True for reco objects that are matched to some gen.
    matched_gen_idx : jagged int [events][reco]
        gen index for each reco. Should be >=0 for matched reco, -1 for nonmatched.
    pt_field : str
        pT field name on reco.

    Returns
    -------
    keep_mask : jagged bool [events][reco]
        True only for the "best" reco per (event, gen).
    """
    n = ak.num(reco, axis=1)  # [events]
    if ak.sum(n) == 0:
        return matched_mask  # empty

    # local reco index within each event: [events][reco]
    local = ak.local_index(reco, axis=1)

    # event id broadcasted to reco shape: [events][reco]
    evt = ak.broadcast_arrays(ak.local_index(n, axis=0), local)[0]

    # offsets per event so we can build a unique global id per reco candidate
    n_np = ak.to_numpy(n)

    offsets_np = np.concatenate(([0], np.cumsum(n_np)[:-1]))
    offsets = ak.Array(offsets_np)

    # global id per reco: [events][reco]
    gid = offsets[:, None] + local

    # Work only with matched reco with valid gen index
    valid = matched_mask & (matched_gen_idx >= 0)

    # Flatten the needed arrays
    gid_f = ak.to_numpy(ak.flatten(gid))
    evt_f = ak.to_numpy(ak.flatten(evt))
    gen_f = ak.to_numpy(ak.flatten(matched_gen_idx))
    pt_f  = ak.to_numpy(ak.flatten(getattr(reco, pt_field)))
    valid_f = ak.to_numpy(ak.flatten(valid))

    if valid_f.sum() == 0:
        # no matched reco anywhere
        return ak.zeros_like(matched_mask, dtype=bool)

    gid_m = gid_f[valid_f]
    evt_m = evt_f[valid_f]
    gen_m = gen_f[valid_f]
    pt_m  = pt_f[valid_f]

    # group key: (event, gen)
    K = 1_000_000
    key = evt_m.astype(np.int64) * K + gen_m.astype(np.int64)

    # Sort by (key asc, pt desc) so first occurrence per key is the best-pt reco
    order = np.lexsort((-pt_m, key))
    key_s = key[order]
    gid_s = gid_m[order]

    # pick the first index for each unique key
    _, first_idx = np.unique(key_s, return_index=True)
    best_gid = gid_s[first_idx]

    # Build a flat keep array over ALL reco candidates in the chunk
    keep_flat = np.zeros_like(gid_f, dtype=bool)
    keep_flat[best_gid] = True

    # Unflatten back to jagged [events][reco]
    keep_mask = ak.unflatten(keep_flat, n)

    return keep_mask


def get_jetConstituents(jet, ptcl, dr_cut=0.4):
    # Get constituents of jets from a collection of particles within dr_cut
    jet_ptcl_pairs = ak.cartesian([jet, ptcl], axis=1, nested=True)
    jet_cands, ptcl_cands = ak.unzip(jet_ptcl_pairs)

    dEta = jet_cands.eta - ptcl_cands.eta
    dPhi = np.abs(jet_cands.phi - ptcl_cands.phi)
    dPhi = ak.where(dPhi > np.pi, 2*np.pi - dPhi, dPhi)
    dR = np.sqrt(dEta**2 + dPhi**2)

    matched_mask = dR < dr_cut
    matched_ptcls = ptcl_cands[matched_mask]
    return matched_ptcls
